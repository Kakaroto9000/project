"""
The (article, company) table for one big model over the UNIVERSE_SIZE most
traded companies (collect/universe.py) -- not only the companies Claude
tagged. Writes data/processed/universe_dataset.parquet (one row per pair) and
universe_articles.parquet (one row per article).

Which pairs: every article x every company would be ~170M rows, almost all
unrelated, so each article keeps only companies with a reason to react:
  - the market/sector funds (INDEX_TICKERS), always
  - companies NAMED in it: their short name ("Nvidia"), or their ticker as
    "(NVDA)", "$NVDA" or "NASDAQ: NVDA"
  - PEERS: the UNIVERSE_PEERS companies whose daily returns move most closely
    with a named one (correlation >= UNIVERSE_PEER_MIN_CORR), e.g. AMD for
    Nvidia news
  - companies the local tagger (modeling/tagger.py) gives >= UNIVERSE_TAGGER_MIN
Only what's available without Claude is used, so the same rule works live.

Inputs per pair, all known when the move starts:
  how related   named_count, named_in_headline, n_named (companies named in
                the article), peer_max / peer_sum (correlation with the named
                companies), tagger_p
  the company   beta to SPY, log dollar volume, cluster (a group of companies
                that move together, from return correlations), plus its own
                pre_news_pct, open_gap_pct, volume_ratio, ret_1d/5d,
                volatility_20d -- the same definitions as modeling/dataset.py
Correlations, betas and clusters use only days before the validation period.

Label: the company's move from entry to exit (market_time.py) minus SPY's
over the same window, cut at FLAT_THRESHOLD_FRACTION of the company's usual
move for that long. Fund rows use their own move. The article's own inputs
(news session, outlet, SPY's move before the news, ...) come from
labeled_dataset.parquet's SPY rows, so both tables share one clock.

Run (after collect/universe.py and modeling/dataset.py):
    python -m newsmodel.modeling.universe_dataset
"""

import json
import re
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from newsmodel import config, market_time
from newsmodel.collect.universe import MINUTE_DIR, UNIVERSE_PATH, latest_daily
from newsmodel.keep_awake import keep_awake
from newsmodel.modeling.dataset import ARTICLES_PATH, DATASET_PATH

PAIRS_PATH = config.PROCESSED_DIR / "universe_dataset.parquet"
ARTICLE_TABLE_PATH = config.PROCESSED_DIR / "universe_articles.parquet"
PAIR_INDEX_PATH = config.PROCESSED_DIR / "universe_pair_index.parquet"  # which companies each article gets
SESSION_MINUTES = 390
MIN = 60 * 10**9  # one minute in nanoseconds

# Words that start many company names, so the first word alone would match the wrong things
GENERIC = set(ENGLISH_STOP_WORDS) | {
    "general", "american", "united", "first", "national", "international", "global", "energy", "capital",
    "financial", "bank", "health", "healthcare", "royal", "new", "digital", "data", "technologies",
    "technology", "systems", "solutions", "services", "resources", "industries", "enterprises",
    "interactive", "entertainment", "communications", "pharmaceuticals", "therapeutics", "sciences",
    "medical", "power", "southern", "northern", "western", "eastern", "pacific", "atlantic",
    "continental", "home", "life", "pure", "real", "west", "east", "north", "south", "world",
    "advanced", "applied", "public", "smart", "super", "true", "union", "vital", "summit", "core",
    # places and nationalities ("Texas Instruments", "Canadian Pacific", "British American")
    "texas", "california", "florida", "york", "boston", "chicago", "dallas", "houston", "phoenix", "denver",
    "atlanta", "carolina", "virginia", "jersey", "washington", "ohio", "michigan", "arizona", "nevada",
    "american", "british", "canadian", "chinese", "japanese", "european", "french", "german", "swiss",
    "australian", "indian", "korean", "taiwan", "mexican", "brazilian", "dutch", "irish", "scottish",
    "america", "canada", "china", "japan", "europe", "india", "korea", "mexico", "brazil", "germany",
    # people's names, days, months, commodities and other everyday words
    "james", "john", "jones", "stanley", "powell", "morgan", "goldman", "wells", "charles", "edward",
    "johnson", "williams", "martin", "live", "smith", "brown", "taylor", "thomas", "lee", "harris", "white",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "june", "july", "august", "september", "october", "november", "december",
    "gold", "silver", "copper", "steel", "iron", "coal", "oil", "natural", "silicon", "solar", "wind", "water",
    "check", "trade", "zoom", "target", "block", "match", "snap", "shift", "booking", "progressive", "sun",
    "state", "street", "main", "park", "rock", "crown", "star", "eagle", "lincoln", "liberty", "pioneer",
}
NAME_JUNK = re.compile(
    r"\b(?:common stock|ordinary shares|american depositary shares|american depositary receipts?|"
    r"depositary shares|class [a-c]|series [a-c]|inc|incorporated|corporation|corp|company|co|ltd|"
    r"limited|plc|n\.?v|s\.?a|ag|se|holdings?|group|the)\b\.?", re.I)
FUND_WORDS = re.compile(r"\b(?:etf|fund|trust|ishares|spdr|vanguard|invesco|proshares|select sector)\b", re.I)
TICKER_IN_TEXT = re.compile(r"(?:\(|\$|(?:NASDAQ|NYSE|Nasdaq|AMEX)\s?:\s?)([A-Z]{1,5}(?:\.[A-Z])?)\b")
WORD = re.compile(r"[A-Za-z0-9&]+")
# Never matched by name (tickers still work): the exchange, a news-footer word, Fed chair Powell
NOT_NAMES = {"nasdaq", "thomson", "thomson reuters", "powell", "new york", "u s", "bank of"}
CAPS_WORD = re.compile(r"[A-Z]{3,5}")
# Names the articles actually use, where the official name's first words don't fit
ALIASES = {
    "TSM": ["tsmc", "taiwan semiconductor"], "MSTR": ["microstrategy"], "GOOGL": ["google", "alphabet"],
    "GOOG": ["google", "alphabet"], "META": ["meta", "facebook"], "BRK.B": ["berkshire"], "KO": ["coca cola"],
    "XOM": ["exxon"], "CVX": ["chevron"], "JNJ": ["j&j"], "PG": ["procter"], "UNH": ["unitedhealth"],
    "BAC": ["bank of america"], "BK": ["bny mellon"], "BNY": ["bny mellon"], "BNS": ["scotiabank"],
    "USB": ["u s bancorp", "us bancorp"], "NYT": ["new york times"], "TRI": [], "GS": ["goldman"], "MS": ["morgan stanley"], "WFC": ["wells fargo"], "C": ["citigroup", "citi"],
    "DIS": ["disney"], "T": ["at&t"], "VZ": ["verizon"], "GM": ["general motors"], "F": ["ford"],
    "BA": ["boeing"], "GE": ["ge aerospace"], "HD": ["home depot"], "MCD": ["mcdonald"], "NKE": ["nike"],
    "SBUX": ["starbucks"], "PFE": ["pfizer"], "MRK": ["merck"], "ABBV": ["abbvie"], "LLY": ["eli lilly", "lilly"],
    "NVO": ["novo nordisk"], "AMD": ["amd"], "IBM": ["ibm"], "SNOW": ["snowflake"], "SHOP": ["shopify"],
    "ARM": ["arm holdings"], "COIN": ["coinbase"], "UBER": ["uber"], "ABNB": ["airbnb"], "DELL": ["dell"],
}


def short_name(name: str):
    """'NVIDIA Corporation Common Stock' -> 'nvidia'; funds get none (matched by ticker only)."""
    if not name or FUND_WORDS.search(name):
        return None
    words = WORD.findall(NAME_JUNK.sub(" ", name.replace(",", " ")))
    if not words:
        return None
    first = words[0].lower()
    if len(first) >= 4 and first not in GENERIC:
        return first
    two = " ".join(w.lower() for w in words[:2]) if len(words) > 1 else None
    return two if two and words[1].lower() not in ENGLISH_STOP_WORDS else None


def ngram_counts(text: str) -> dict:
    """How often each phrase appears: two-word phrases in any case, single words only when
    capitalized, so "price target" doesn't count as Target but "Target said" does."""
    raw = WORD.findall(text)
    tokens = [t.lower() for t in raw]
    counts = {}
    for t, low in zip(raw, tokens):
        if t[0].isupper():
            counts[low] = counts.get(low, 0) + 1
    for n in (2, 3):
        for i in range(len(tokens) - n + 1):
            g = " ".join(tokens[i:i + n])
            counts[g] = counts.get(g, 0) + 1
    return counts


_NAMES = _TICKERS = None


def _init_matcher(names, tickers):
    global _NAMES, _TICKERS
    _NAMES, _TICKERS = names, tickers


def _match(args):
    """For one article: {company index: (times named, named in headline)}."""
    headline, body = args
    found = {}
    for text, in_headline in ((headline, True), (body, False)):
        counts = ngram_counts(text)
        hits = []
        for phrase, idxs in _NAMES.items():
            c = counts.get(phrase)
            if c:
                hits += [(i, c) for i in idxs]
        # Tickers written as "(NVDA)", "$NVDA", "NASDAQ: NVDA", or plain caps if not an English word
        tickers = TICKER_IN_TEXT.findall(text) + [t for t in CAPS_WORD.findall(text) if t.lower() not in GENERIC]
        hits += [(_TICKERS[t], 1) for t in tickers if t in _TICKERS]
        for idx, c in hits:
            n, h = found.get(idx, (0, False))
            found[idx] = (n + c, h or in_headline)
    return found


def named_companies(articles: pd.DataFrame, universe: list) -> list:
    names = {}  # phrase -> companies it names (GOOG and GOOGL share "alphabet")
    for i, u in enumerate(universe):
        phrases = ALIASES[u["ticker"]] if u["ticker"] in ALIASES else [short_name(u["name"])]
        for s in phrases:
            if s and s not in NOT_NAMES:
                names.setdefault(s, []).append(i)
    tickers = {u["ticker"]: i for i, u in enumerate(universe)}
    print(f"Matching {len(names)} company names and {len(tickers)} tickers in {len(articles)} articles...", flush=True)
    work = list(zip(articles["headline"].fillna(""), articles["article_text"].fillna("")))
    with Pool(16, initializer=_init_matcher, initargs=(names, tickers)) as pool:
        return pool.map(_match, work, chunksize=500)


def company_traits(daily: pd.DataFrame, tickers: list, before) -> dict:
    """Return correlations, beta to SPY, dollar volume and a cluster id, from days before `before`."""
    d = daily[daily["symbol"].isin(tickers) & (daily["date"] < before)]
    closes = d.pivot_table(index="date", columns="symbol", values="close").reindex(columns=tickers)
    returns = closes.pct_change(fill_method=None).iloc[1:]
    corr = returns.corr(min_periods=60).fillna(0).to_numpy(np.float32)
    np.fill_diagonal(corr, 0)
    spy = returns[config.MARKET_TICKER]
    beta = (returns.apply(lambda r: r.cov(spy)) / spy.var()).fillna(1).to_numpy(np.float32)
    dv = (d["close"] * d["volume"]).groupby(d["symbol"]).mean().reindex(tickers)
    clusters = KMeans(n_clusters=config.UNIVERSE_CLUSTERS, n_init=10, random_state=0).fit_predict(corr)
    return {"corr": corr, "beta": beta, "log_dollar_volume": np.log10(dv.fillna(dv.median())).to_numpy(np.float32),
            "cluster": clusters}


def choose_pairs(found: list, corr: np.ndarray, tagger_p: np.ndarray, tagger_cols: np.ndarray, fund_idx: list):
    """Per article, the company indices to keep and their how-related numbers."""
    rows = {k: [] for k in ("article_idx", "company_idx", "named_count", "named_in_headline",
                            "n_named", "peer_max", "peer_sum", "tagger_p")}
    n_comp = corr.shape[0]
    for a, f in enumerate(found):
        named = np.array(sorted(f), dtype=np.int64)
        keep = set(fund_idx) | set(named.tolist())
        peer_max = np.zeros(n_comp, np.float32)
        peer_sum = np.zeros(n_comp, np.float32)
        if len(named):
            c = corr[:, named]
            peer_max = c.max(axis=1)
            peer_sum = np.clip(c, 0, None).sum(axis=1)
            cand = np.flatnonzero(peer_max >= config.UNIVERSE_PEER_MIN_CORR)
            cand = cand[np.argsort(-peer_max[cand])][:config.UNIVERSE_PEERS]
            keep |= set(cand.tolist())
        tp = np.zeros(n_comp, np.float32)
        if tagger_p is not None:
            tp[tagger_cols] = tagger_p[a]
            keep |= set(np.flatnonzero(tp >= config.UNIVERSE_TAGGER_MIN).tolist())
        for c_idx in sorted(keep):
            n, h = f.get(c_idx, (0, False))
            rows["article_idx"].append(a)
            rows["company_idx"].append(c_idx)
            rows["named_count"].append(n)
            rows["named_in_headline"].append(h)
            rows["n_named"].append(len(named))
            rows["peer_max"].append(peer_max[c_idx])
            rows["peer_sum"].append(peer_sum[c_idx])
            rows["tagger_p"].append(tp[c_idx])
    return pd.DataFrame(rows)


def tagger_scores(articles: pd.DataFrame, universe_index: dict):
    from newsmodel.modeling import tagger
    if not tagger.TAGGER_PATH.exists():
        print("No local tagger saved -- tagger_p will be 0.")
        return None, None
    model = tagger.load()
    cols = [(j, universe_index[t]) for j, t in enumerate(model["tickers"]) if t in universe_index]
    texts = (articles["headline"].fillna("") + ". " + articles["article_text"].fillna("")).tolist()
    P = tagger.probabilities(model, model["vectorizer"].transform(texts)).astype(np.float32)
    print(f"Tagger scores for {len(cols)} of its {len(model['tickers'])} companies that are in the universe")
    return P[:, [j for j, _ in cols]], np.array([u for _, u in cols])


def rolling_daily(daily_one: pd.DataFrame) -> dict:
    """One company's daily series and the per-day numbers the inputs need."""
    d = daily_one.sort_values("timestamp")
    trade_date = pd.to_datetime(d["date"])
    close = d["close"].to_numpy(np.float64)
    ret = pd.Series(close).pct_change()
    return {
        "dates": d["date"].to_numpy(),
        "close_time": ((trade_date + market_time.MARKET_CLOSE).dt.tz_localize(market_time.MARKET_TZ)
                       .dt.tz_convert("UTC").dt.tz_localize(None).to_numpy("datetime64[ns]").astype(np.int64)),
        "open": d["open"].to_numpy(np.float64), "close": close,
        "ret1": ret.to_numpy() * 100,
        "ret5": (pd.Series(close).pct_change(5)).to_numpy() * 100,
        "vol20": (ret.rolling(config.VOLATILITY_DAYS, min_periods=6).std()).to_numpy() * 100,
        "avgvol": pd.Series(d["volume"].to_numpy(np.float64)).rolling(config.VOLATILITY_DAYS, min_periods=1).mean().to_numpy(),
    }


def price_at(t, bar_end, bar_close, close_time, close):
    """Last price known at each time in t: the latest finished minute bar or daily close."""
    i = np.searchsorted(bar_end, t, side="right") - 1
    j = np.searchsorted(close_time, t, side="right") - 1
    ti = np.where(i >= 0, bar_end[np.clip(i, 0, None)] if len(bar_end) else -1, -1)
    tj = np.where(j >= 0, close_time[np.clip(j, 0, None)], -1)
    use_bar = (ti > tj) & (i >= 0)
    p = np.where(use_bar, bar_close[np.clip(i, 0, None)] if len(bar_close) else np.nan,
                 np.where(j >= 0, close[np.clip(j, 0, None)], np.nan))
    return p, np.maximum(ti, tj)


def company_inputs(pairs_c: pd.DataFrame, art: pd.DataFrame, dd: dict, bars: dict, is_fund: bool) -> pd.DataFrame:
    a = pairs_c["article_idx"].to_numpy()
    news = art["news_ns"].to_numpy()[a]
    entry = art["entry_ns"].to_numpy()[a]
    exit_ = art["exit_ns"].to_numpy()[a]
    entry_open = art["entry_open_ns"].to_numpy()[a]
    at_news = art["at_news"].to_numpy()[a]
    minutes = art["label_minutes"].to_numpy()[a]
    pre = config.PRE_NEWS_MINUTES * MIN

    be, bc, bs, cum = bars["end"], bars["close"], bars["start"], bars["cumvol"]
    ct, close = dd["close_time"], dd["close"]
    p_news, _ = price_at(news, be, bc, ct, close)
    p_start, _ = price_at(news - pre, be, bc, ct, close)
    k_entry = np.searchsorted(ct, entry_open, side="left")  # the entry day's daily bar
    ke = np.clip(k_entry, 0, len(ct) - 1)
    entry_day_ok = (k_entry < len(ct)) & (ct[ke] > entry_open) & (ct[ke] - entry_open < 24 * 60 * MIN)
    entry_price = np.where(at_news, p_news, np.where(entry_day_ok, dd["open"][ke], np.nan))
    # The label's end must be a real 1-minute bar on the entry day -- never a daily close,
    # which would measure the whole day instead of HORIZON_MINUTES
    i_exit = np.searchsorted(be, exit_, side="right") - 1
    exit_from_bar = (i_exit >= 0) & (be[np.clip(i_exit, 0, None)] > entry_open) if len(be) else np.zeros(len(a), bool)
    exit_price = np.where(exit_from_bar, bc[np.clip(i_exit, 0, None)] if len(bc) else np.nan, np.nan)
    ok = entry_day_ok & np.isfinite(entry_price) & np.isfinite(exit_price)

    k = np.searchsorted(ct, news, side="right")  # daily closes known at the news
    kk = np.clip(k - 1, 0, len(ct) - 1)
    has = k >= 1
    vol20 = np.where(has, dd["vol20"][kk], np.nan)
    gap = np.where((ke >= 1) & entry_day_ok, (dd["open"][ke] / close[np.clip(ke - 1, 0, None)] - 1) * 100, np.nan)

    # Volume in the pre-news window, for news during market hours
    vol_from = np.maximum(news - pre, art["news_day_open_ns"].to_numpy()[a])
    in_hours = art["market_hours"].to_numpy()[a] & (news - vol_from >= MIN)
    if len(bs):
        lo = np.searchsorted(bs, vol_from, side="left")
        hi = np.searchsorted(be, news, side="right")
        vol = cum[hi] - cum[lo]
    else:
        vol = np.zeros(len(a))
    usual = np.where(has, dd["avgvol"][kk], np.nan) * (news - vol_from) / MIN / SESSION_MINUTES
    volume_ratio = np.where(in_hours & (usual > 0), np.log((vol + 1) / (usual + 1)), np.nan)

    move = (exit_price / entry_price - 1) * 100
    market = art["market_pct_change"].to_numpy()[a]
    trade_move = move if is_fund else move - market
    usual_move = vol20 * np.sqrt(minutes / SESSION_MINUTES)
    threshold = np.where(np.isfinite(usual_move), config.FLAT_THRESHOLD_FRACTION * usual_move, config.FLAT_THRESHOLD_PCT)
    label = np.where(trade_move > threshold, "up", np.where(trade_move < -threshold, "down", "flat"))

    out = pairs_c.assign(
        pre_news_pct=(p_news / p_start - 1) * 100, open_gap_pct=gap, volume_ratio=volume_ratio,
        ret_1d_pct=np.where(has, dd["ret1"][kk], np.nan), ret_5d_pct=np.where(has, dd["ret5"][kk], np.nan),
        volatility_20d_pct=vol20, flat_threshold_pct=threshold,
        entry_price=entry_price, exit_price=exit_price, pct_change=move, trade_move=trade_move,
        hedged=not is_fund, label=label)
    return out[ok]


def load_minute_chunk(symbols: set) -> pd.DataFrame:
    frames = []
    for f in sorted(MINUTE_DIR.glob("*.parquet")):
        df = pd.read_parquet(f)
        frames.append(df[df["symbol"].isin(symbols)].assign(symbol=lambda x: x["symbol"].astype(str)))
    return pd.concat(frames, ignore_index=True)


def article_table() -> pd.DataFrame:
    """One row per article: its clock and its own inputs, from labeled_dataset.parquet's SPY rows."""
    spy = pd.read_parquet(DATASET_PATH)
    spy = spy[spy["ticker"] == config.MARKET_TICKER].sort_values("news_timestamp").reset_index(drop=True)
    to_ns = lambda s: pd.to_datetime(s, utc=True).dt.tz_localize(None).to_numpy("datetime64[ns]").astype(np.int64)
    entry_day = spy["entry_timestamp"].map(market_time.market_date)
    news_day = spy["news_timestamp"].map(market_time.market_date)
    art = pd.DataFrame({
        "article_id": spy["article_id"].values,
        "news_timestamp": spy["news_timestamp"],  # keeps the UTC time zone (.values would drop it)
        "news_ns": to_ns(spy["news_timestamp"]), "entry_ns": to_ns(spy["entry_timestamp"]),
        "exit_ns": to_ns(spy["exit_timestamp"]),
        "entry_open_ns": to_ns(entry_day.map(market_time.session_open)),
        "news_day_open_ns": to_ns(news_day.map(market_time.session_open)),
        "at_news": (spy["entry"] == "at_news").values,
        "market_hours": (spy["news_session"] == "market_hours").values,
        "market_pct_change": spy["pct_change"].values,
    })
    for col in ("news_session", "entry", "hours_to_entry", "label_minutes", "source_name", "day_of_week",
                "n_tickers", "spy_pre_news_pct", "spy_open_gap_pct"):
        art[col] = spy[col].values
    return art


def build():
    t0 = time.time()
    with keep_awake():
        art = article_table()
        texts = pd.read_parquet(ARTICLES_PATH).set_index("article_id").reindex(art["article_id"]).reset_index()
        universe = json.loads(UNIVERSE_PATH.read_text())
        tickers = [u["ticker"] for u in universe]
        index = {t: i for i, t in enumerate(tickers)}
        fund_idx = [index[t] for t in config.INDEX_TICKERS if t in index]

        daily = latest_daily()
        daily = daily[daily["symbol"].isin(tickers)]
        val_start = art["news_timestamp"].iloc[int(len(art) * config.TRAIN_FRACTION)]
        traits = company_traits(daily, tickers, market_time.market_date(pd.Timestamp(val_start)))
        print(f"Company traits from days before {market_time.market_date(pd.Timestamp(val_start))} ({time.time() - t0:.0f}s)", flush=True)

        if PAIR_INDEX_PATH.exists():  # matching is the slow part and doesn't depend on prices
            pairs = pd.read_parquet(PAIR_INDEX_PATH)
            print(f"Reusing the article-company pairing in {PAIR_INDEX_PATH.name} (delete it to redo)", flush=True)
        else:
            found = named_companies(texts, universe)
            print(f"Named-company matching done ({time.time() - t0:.0f}s); articles naming at least one: "
                  f"{sum(1 for f in found if f) / len(found):.0%}", flush=True)
            tagger_p, tagger_cols = tagger_scores(texts, index)
            pairs = choose_pairs(found, traits["corr"], tagger_p, tagger_cols, fund_idx)
            pairs.to_parquet(PAIR_INDEX_PATH, index=False)
        print(f"{len(pairs)} (article, company) pairs, {len(pairs) / len(art):.1f} per article ({time.time() - t0:.0f}s)", flush=True)

        out = []
        daily_by = dict(tuple(daily.groupby("symbol")))
        chunk = 250
        for c0 in range(0, len(tickers), chunk):
            syms = tickers[c0:c0 + chunk]
            minute = load_minute_chunk(set(syms))
            minute_by = dict(tuple(minute.groupby("symbol")))
            for sym in syms:
                ci = index[sym]
                pc = pairs[pairs["company_idx"] == ci]
                if pc.empty or sym not in daily_by:
                    continue
                m = minute_by.get(sym)
                if m is not None and len(m):
                    m = m.sort_values("timestamp")
                    start = pd.to_datetime(m["timestamp"], utc=True).dt.tz_localize(None).to_numpy("datetime64[ns]").astype(np.int64)
                    bars = {"start": start, "end": start + config.INTRADAY_BAR_MINUTES * MIN,
                            "close": m["close"].to_numpy(np.float64),
                            "cumvol": np.concatenate([[0], np.cumsum(m["volume"].to_numpy(np.float64))])}
                else:
                    bars = {"start": np.array([], np.int64), "end": np.array([], np.int64),
                            "close": np.array([]), "cumvol": np.array([0.0])}
                out.append(company_inputs(pc, art, rolling_daily(daily_by[sym]), bars, sym in config.INDEX_TICKERS))
            del minute, minute_by
            print(f"  prices done for {min(c0 + chunk, len(tickers))}/{len(tickers)} companies ({time.time() - t0:.0f}s)", flush=True)

        ds = pd.concat(out, ignore_index=True)
        ds["ticker"] = np.array(tickers)[ds["company_idx"]]
        ds["is_fund"] = ds["ticker"].isin(config.INDEX_TICKERS)
        for name in ("beta", "log_dollar_volume", "cluster"):
            ds[name] = traits[name][ds["company_idx"]]
        ds = ds.sort_values(["article_idx", "company_idx"]).reset_index(drop=True)
        ds.to_parquet(PAIRS_PATH, index=False)
        art.drop(columns=[c for c in art.columns if c.endswith("_ns")]).to_parquet(ARTICLE_TABLE_PATH, index=False)

    print(f"\nSaved {len(ds)} pairs to {PAIRS_PATH} ({time.time() - t0:.0f}s)")
    print("Label shares:", ds["label"].value_counts(normalize=True).round(3).to_dict())
    print("Pairs by kind: funds", int(ds["is_fund"].sum()), "| named", int((ds["named_count"] > 0).sum()),
          "| peers/tagger only", int(((ds["named_count"] == 0) & ~ds["is_fund"]).sum()))
    print("Most-paired companies:", ds.loc[~ds["is_fund"], "ticker"].value_counts().head(12).to_dict())


if __name__ == "__main__":
    build()
