"""
The article x stock grid for the one model that predicts all UNIVERSE_SIZE stocks
from each article (modeling/news_model.py). No tickers or tags: every article gets
an answer for every stock, and the model learns which ones it actually moves.

Saved as arrays in data/processed/grid/ (article order = universe_articles.parquet,
stock order = data/raw_prices/universe.json):
  y_main     (articles, stocks) int8   the label traded on: entry -> exit
                                       (market_time.py), vs. SPY for stocks, own
                                       move for INDEX_TICKERS; 0 down, 1 flat,
                                       2 up, -1 no price
  y_aux      (articles, stocks) int8   practice label, training only: 10 minutes
                                       before the article -> exit, same rules
  move       (articles, stocks) f16    the traded move in % (for the backtest)
  prices     (articles, stocks, 6) f16 known before the news: last hour's move,
                                       opening jump, volume vs. usual, 1- and
                                       5-day return, usual daily move
  article    (articles, 9) f32         SPY's last hour and opening jump, hours the
                                       news waited, window length, traded at the
                                       news or the next open, news session (4)
The "up"/"down" cutoff is per stock: FLAT_THRESHOLD_FRACTION of its usual move
over the window, as everywhere else.

Run (after collect/universe.py and modeling/universe_dataset.py):
    python -m newsmodel.modeling.grid_dataset
"""

import json
import time

import numpy as np
import pandas as pd

from newsmodel import config
from newsmodel import market_time
from newsmodel.collect.universe import EXT_MINUTE_DIR, MINUTE_DIR, UNIVERSE_PATH, latest_daily
from newsmodel.keep_awake import keep_awake
from newsmodel.modeling.universe_dataset import MIN, SESSION_MINUTES, article_table, price_at, rolling_daily

GRID_DIR = config.PROCESSED_DIR / "grid"          # regular hours only (traded at the next open when closed)
EXT_GRID_DIR = config.PROCESSED_DIR / "grid_ext"  # with pre-market and after-hours trading
AUX_LEAD_MINUTES = 10
ENTRY_MAX_WAIT_MINUTES = 5  # extended hours: the stock must trade this soon after the window opens
PRICE_NAMES = ["pre_news_pct", "open_gap_pct", "volume_ratio", "ret_1d_pct", "ret_5d_pct", "volatility_20d_pct"]
ARTICLE_NAMES = ["spy_pre_news_pct", "spy_open_gap_pct", "hours_to_entry", "label_minutes", "at_news",
                 "pre_market", "market_hours", "after_hours", "closed_day"]


def stock_columns(art, dd, bars, extended=False):
    """Every article's entry/exit prices, moves and inputs for one stock (NaN where unknown)."""
    news, exit_ = art["news_ns"].to_numpy(), art["exit_ns"].to_numpy()
    entry_open, at_news = art["entry_open_ns"].to_numpy(), art["at_news"].to_numpy()
    pre = config.PRE_NEWS_MINUTES * MIN
    be, bc, bs, cum = bars["end"], bars["close"], bars["start"], bars["cumvol"]
    ct, close = dd["close_time"], dd["close"]

    p_news, _ = price_at(news, be, bc, ct, close)
    p_start, _ = price_at(news - pre, be, bc, ct, close)
    p_lead, _ = price_at(news - AUX_LEAD_MINUTES * MIN, be, bc, ct, close)
    k_entry = np.searchsorted(ct, entry_open, side="left")
    ke = np.clip(k_entry, 0, len(ct) - 1)
    day_ok = (k_entry < len(ct)) & (ct[ke] > entry_open) & (ct[ke] - entry_open < 24 * 60 * MIN)
    if extended:
        # Entry = the FIRST real trade at or after the window opens -- a price you could actually
        # get. The last trade before it can be hours or a weekend old for a thinly traded stock
        # (Friday's price for Monday 4:00 news), which would put an untradeable gap in the label.
        # No trade within ENTRY_MAX_WAIT_MINUTES -> you couldn't have traded it: no label.
        entry_ns = art["entry_ns"].to_numpy()
        if len(bs):
            j = np.searchsorted(bs, entry_ns, side="left")
            jj = np.clip(j, 0, len(bs) - 1)
            traded = (j < len(bs)) & (bs[jj] - entry_ns <= ENTRY_MAX_WAIT_MINUTES * MIN) & (entry_ns > 0)
            entry = np.where(traded, bc[jj], np.nan)
            after = np.where(traded, be[jj], np.iinfo(np.int64).max)  # the exit must be a later trade
        else:
            entry, after = np.full(len(news), np.nan), np.full(len(news), np.iinfo(np.int64).max)
        day_ok = entry_ns > 0
    else:
        entry = np.where(at_news, p_news, np.where(day_ok, dd["open"][ke], np.nan))
        after = entry_open
    if len(be):
        i = np.searchsorted(be, exit_, side="right") - 1
        from_bar = (i >= 0) & (be[np.clip(i, 0, None)] > after)  # a trade between entry and exit
        exit_p = np.where(from_bar, bc[np.clip(i, 0, None)], np.nan)
    else:
        exit_p = np.full(len(news), np.nan)
    ok = day_ok & np.isfinite(entry) & np.isfinite(exit_p)

    # The day's 9:30 open is only known once the window starts at or after 9:30. For a
    # pre-market window (e.g. 4:00-4:30) it's the future, and pre-market prices drift
    # toward it, so using it would leak the answer.
    gap_known = (art["entry_ns"].to_numpy() >= entry_open) if extended else np.ones(len(news), bool)
    k = np.searchsorted(ct, news, side="right")
    kk = np.clip(k - 1, 0, len(ct) - 1)
    has = k >= 1
    vol_from = np.maximum(news - pre, art["news_day_open_ns"].to_numpy())
    in_hours = art["market_hours"].to_numpy() & (news - vol_from >= MIN)
    if len(bs):
        vol = cum[np.searchsorted(be, news, side="right")] - cum[np.searchsorted(bs, vol_from, side="left")]
    else:
        vol = np.zeros(len(news))
    usual = np.where(has, dd["avgvol"][kk], np.nan) * (news - vol_from) / MIN / SESSION_MINUTES
    prices = np.stack([
        (p_news / p_start - 1) * 100,
        np.where((ke >= 1) & day_ok & gap_known, (dd["open"][ke] / close[np.clip(ke - 1, 0, None)] - 1) * 100, np.nan),
        np.where(in_hours & (usual > 0), np.log((vol + 1) / (usual + 1)), np.nan),
        np.where(has, dd["ret1"][kk], np.nan),
        np.where(has, dd["ret5"][kk], np.nan),
        np.where(has, dd["vol20"][kk], np.nan),
    ], axis=1)
    main = np.where(ok, (exit_p / entry - 1) * 100, np.nan)
    aux = np.where(ok & np.isfinite(p_lead), (exit_p / p_lead - 1) * 100, np.nan)
    return main, aux, prices


def to_label(move, threshold):
    y = np.where(move > threshold, 2, np.where(move < -threshold, 0, 1)).astype(np.int8)
    y[~np.isfinite(move)] = -1
    return y


def ext_windows(art, trading_days):
    """Extended-hours entry and exit for every article (market_time.ext_entry_time)."""
    to_ns = lambda ts: ts.tz_convert("UTC").tz_localize(None).value
    entry, exit_, regular, day_open = [], [], [], []
    for t in art["news_timestamp"]:
        e = market_time.ext_entry_time(pd.Timestamp(t), trading_days)
        if e is None:
            entry.append(-1); exit_.append(-1); regular.append(False); day_open.append(-1); continue
        x = market_time.ext_exit_for_entry(e)
        entry.append(to_ns(e)); exit_.append(to_ns(x)); regular.append(market_time.is_regular_window(e, x))
        day_open.append(to_ns(market_time.session_open(market_time.market_date(e))))
    return np.array(entry, np.int64), np.array(exit_, np.int64), np.array(day_open, np.int64), np.array(regular)


def load_minutes(symbols: set, extended: bool) -> pd.DataFrame:
    from newsmodel.modeling.universe_dataset import load_minute_chunk
    df = load_minute_chunk(symbols)
    if extended:
        frames = [df]
        for f in sorted(EXT_MINUTE_DIR.glob("*.parquet")):
            x = pd.read_parquet(f)
            frames.append(x[x["symbol"].isin(symbols)].assign(symbol=lambda y: y["symbol"].astype(str)))
        df = pd.concat(frames, ignore_index=True)
    return df


def grid_dir(extended: bool, horizon: int = None):
    """grid/ or grid_ext/ for the usual HORIZON_MINUTES; grid_ext_<minutes>/ for another window length."""
    base = EXT_GRID_DIR if extended else GRID_DIR
    h = horizon or config.HORIZON_MINUTES
    return base if h == 30 else base.parent / f"{base.name}_{h}"


def build(extended=False):
    t0 = time.time()
    out_dir = grid_dir(extended)
    with keep_awake():
        art = article_table()
        universe = [u["ticker"] for u in json.loads(UNIVERSE_PATH.read_text())]
        daily = latest_daily()
        daily = daily[daily["symbol"].isin(universe)]
        daily_by = dict(tuple(daily.groupby("symbol")))
        regular = np.ones(len(art), bool)
        if extended:
            trading_days = sorted(daily.loc[daily["symbol"] == config.MARKET_TICKER, "date"].unique())
            entry_ns, exit_ns, open_ns, regular = ext_windows(art, trading_days)
            art["entry_ns"], art["exit_ns"], art["entry_open_ns"] = entry_ns, exit_ns, open_ns
            art["at_news"] = entry_ns <= art["news_ns"].to_numpy() + MIN  # traded right at the news
            art["label_minutes"] = np.where(entry_ns > 0, (exit_ns - entry_ns) / MIN, np.nan)
            art["hours_to_entry"] = np.where(entry_ns > 0, (entry_ns - art["news_ns"].to_numpy()) / (60 * MIN), np.nan)
            print(f"Extended-hours windows: {np.mean(entry_ns > 0):.0%} of articles placed, "
                  f"{regular.mean():.0%} fully inside 9:30-16:00 ({time.time() - t0:.0f}s)", flush=True)
        n_a, n_s = len(art), len(universe)
        y_main = np.full((n_a, n_s), -1, np.int8)
        y_aux = np.full((n_a, n_s), -1, np.int8)
        move = np.full((n_a, n_s), np.nan, np.float16)
        prices = np.full((n_a, n_s, len(PRICE_NAMES)), np.nan, np.float16)
        minutes = art["label_minutes"].to_numpy()

        raw = {}  # move -> per stock, before subtracting SPY
        index = {t: i for i, t in enumerate(universe)}
        for c0 in range(0, n_s, 250):
            syms = universe[c0:c0 + 250]
            minute = load_minutes(set(syms), extended)
            minute_by = dict(tuple(minute.groupby("symbol")))
            for sym in syms:
                if sym not in daily_by:
                    continue
                m = minute_by.get(sym)
                if m is not None and len(m):
                    m = m.sort_values("timestamp")
                    start = pd.to_datetime(m["timestamp"], utc=True).dt.tz_localize(None).to_numpy("datetime64[ns]").astype(np.int64)
                    bars = {"start": start, "end": start + config.INTRADAY_BAR_MINUTES * MIN, "close": m["close"].to_numpy(np.float64),
                            "cumvol": np.concatenate([[0], np.cumsum(m["volume"].to_numpy(np.float64))])}
                else:
                    bars = {"start": np.array([], np.int64), "end": np.array([], np.int64), "close": np.array([]), "cumvol": np.array([0.0])}
                dd = rolling_daily(daily_by[sym])
                main, aux, p = stock_columns(art, dd, bars, extended)
                raw[sym] = (main.astype(np.float32), aux.astype(np.float32), p[:, 5].astype(np.float32))
                prices[:, index[sym]] = p.astype(np.float16)
            del minute, minute_by
            print(f"  prices for {min(c0 + 250, n_s)}/{n_s} stocks ({time.time() - t0:.0f}s)", flush=True)

        spy_main, spy_aux, _ = raw[config.MARKET_TICKER]
        for sym, (main, aux, vol20) in raw.items():
            j = index[sym]
            fund = sym in config.INDEX_TICKERS
            m = main if fund else main - spy_main
            a = aux if fund else aux - spy_aux
            usual = vol20 * np.sqrt(minutes / SESSION_MINUTES)
            thr = np.where(np.isfinite(usual), config.FLAT_THRESHOLD_FRACTION * usual, config.FLAT_THRESHOLD_PCT)
            thr_aux = thr * np.sqrt((minutes + AUX_LEAD_MINUTES) / minutes)
            y_main[:, j] = to_label(m, thr)
            y_aux[:, j] = to_label(a, thr_aux)
            move[:, j] = m.astype(np.float16)

        session = art["news_session"].to_numpy()
        spy_gap = art["spy_open_gap_pct"].to_numpy().astype(np.float64)
        if extended:  # SPY's opening jump is the future for windows that start before 9:30
            spy_gap = np.where(art["entry_ns"].to_numpy() >= art["entry_open_ns"].to_numpy(), spy_gap, np.nan)
        article = np.stack([art["spy_pre_news_pct"].to_numpy(), spy_gap, art["hours_to_entry"].to_numpy(),
                            minutes, art["at_news"].to_numpy(), session == "pre_market", session == "market_hours",
                            session == "after_hours", np.isin(session, ["weekend", "holiday"])], axis=1).astype(np.float32)
        # Each 30-minute window counts once in training: articles sharing one split its weight
        win = pd.Series(art["entry_ns"].to_numpy())
        weight = (1.0 / win.map(win.value_counts())).to_numpy(np.float32)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, arr in (("y_main", y_main), ("y_aux", y_aux), ("move", move), ("prices", prices), ("article", article),
                          ("weight", weight), ("regular", regular)):
            np.save(out_dir / f"{name}.npy", arr)
        (out_dir / "meta.json").write_text(json.dumps({"stocks": universe, "price_names": PRICE_NAMES, "article_names": ARTICLE_NAMES,
                                                        "funds": [t for t in config.INDEX_TICKERS if t in index]}, indent=1))

    have = y_main >= 0
    print(f"\nSaved the {n_a} x {n_s} grid to {out_dir} ({time.time() - t0:.0f}s)")
    print(f"Labels known for {have.mean():.0%} of article-stock cells")
    print("Label shares:", {k: round(float((y_main[have] == i).mean()), 3) for i, k in enumerate(["down", "flat", "up"])})


if __name__ == "__main__":
    import sys
    if "--horizon" in sys.argv:  # a different window length, saved to its own folder
        config.HORIZON_MINUTES = int(sys.argv[sys.argv.index("--horizon") + 1])
    build(extended="--extended" in sys.argv)
