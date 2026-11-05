"""
Combine tagged news + prices into a single labeled training table
(data/processed/labeled_dataset.parquet). First step of Job 3.

Reads data/processed/news_tagged.json (Job 2's output) and the latest
data/raw_prices/prices_daily_*.parquet and prices_intraday_*.parquet
(collect/prices.py).

For each (article, ticker) pair, with T = when the news came out (config.NEWS_TIME_FIELD,
the outlet's publish time by default):
  1. Label: the HORIZON_MINUTES move from the first moment you could trade
     on the news (examples in market_time.py). During market hours it
     starts from the price at T; while the market is closed it starts from
     the next day's official opening price, so the overnight/weekend jump
     isn't counted. News in the last HORIZON_MINUTES of a session is
     measured to the 16:00 close instead (label_minutes is the window's
     length). "up" / "down" / "flat" -- always one of these three, never a
     percentage. The cutoff where "flat" starts is per stock --
     FLAT_THRESHOLD_FRACTION of its usual move over the window -- or one
     fixed FLAT_THRESHOLD_PCT (see FLAT_THRESHOLD_SCALED in config.py).
     Each row's cutoff is stored in flat_threshold_pct.
     "Price at" a time is the last price known by then: the last finished
     intraday bar, or the last daily close (known at 4pm New York time) if
     that's more recent.
  2. What the model sees -- all known when the move starts:
       pre_news_pct        price PRE_NEWS_MINUTES before T -> price at T
       spy_pre_news_pct    the same move for MARKET_TICKER, so a stock's
                           own move can be told from the market's
       open_gap_pct,       previous close -> opening price on the day the
       spy_open_gap_pct    move starts. For news while the market is closed
                           that's how much the news is already priced in
       volume_ratio        shares traded in that window vs. the stock's
                           usual volume for that long (log scale: 0 =
                           usual, 1 = ~2.7x usual). Only for news during
                           market hours; missing otherwise.
       ret_1d_pct,         the stock's return over the last 1 and 5
       ret_5d_pct          trading days before T
       volatility_20d_pct  its typical daily move over the 20 days before
  3. news_session: when the article came out relative to the US market --
     "weekend", "holiday", "pre_market", "market_hours" or "after_hours".
     entry: when the trade happens -- "at_news" or "next_open" (news while
     the market was closed, or with under LATE_NEWS_MIN_MINUTES left in a
     session, which news_session alone calls market_hours). hours_to_entry:
     how long the news waited before the trade (0 at_news, ~65 for
     Friday-evening news). label_minutes: how long the predicted move lasts
     (HORIZON_MINUTES, less for news just before the close).
  4. role: how the news reaches this company, from Job 2 -- "direct",
     "competitor", "opposite_market", "supplier_customer", "world_event",
     "policy_macro" or "other" (the list is ROLES in collect/tickers.py).
     The same article can lift the company it's about and sink its
     competitor.
  5. Article details: source_name (the outlet), n_tickers (companies the
     article affects), ticker_rank (1 = most affected, per Job 2) and
     day_of_week.
  6. "headline" + "article_text" (Event Registry's full body) are the text
     the model trains on. They're saved once per article in
     data/processed/articles.parquet, not in every row -- one article has a
     row per company and per fund, so copying its text into each would make
     the dataset ~10x bigger. Rows point to their article by article_id. "sentiment_score" (Event Registry's own -1..1
     float, 0.0 if missing) is stored for reference only.

Run (normally via job3-train):
    python -m newsmodel.modeling.dataset
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from newsmodel import config, market_time

DATASET_PATH = config.PROCESSED_DIR / "labeled_dataset.parquet"
ARTICLES_PATH = config.PROCESSED_DIR / "articles.parquet"  # each article's text, once
SESSION_MINUTES = 390  # 9:30 to 16:00


def load_latest(directory: Path, pattern: str) -> pd.DataFrame:
    files = sorted(directory.glob(pattern))
    if not files:
        raise SystemExit(f"No files matching {pattern} in {directory}. Run python -m newsmodel.collect.prices first.")
    return pd.read_parquet(files[-1])


def _ns(times: pd.Series) -> np.ndarray:
    """UTC timestamps as int64 nanoseconds, for fast searchsorted."""
    return pd.to_datetime(times, utc=True).dt.tz_localize(None).to_numpy("datetime64[ns]").astype("int64")


def pct(start, end):
    return (end - start) / start * 100 if start and end is not None else np.nan


class TickerPrices:
    """One ticker's prices, for looking up the last price known at any moment."""

    def __init__(self, daily: pd.DataFrame, intraday: pd.DataFrame):
        daily = daily.sort_values("timestamp")
        trade_date = pd.to_datetime(pd.to_datetime(daily["timestamp"], utc=True)
                                    .dt.tz_convert(market_time.MARKET_TZ).dt.date)
        self.daily_dates = trade_date.dt.date.to_numpy()
        self.close_time = _ns((trade_date + market_time.MARKET_CLOSE).dt.tz_localize(market_time.MARKET_TZ))
        self.daily_open = daily["open"].to_numpy(float)
        self.daily_close = daily["close"].to_numpy(float)
        self.daily_volume = daily["volume"].to_numpy(float)

        intraday = intraday.sort_values("timestamp")
        self.bar_start = _ns(intraday["timestamp"])
        self.bar_end = self.bar_start + pd.Timedelta(minutes=config.INTRADAY_BAR_MINUTES).value
        self.bar_close = intraday["close"].to_numpy(float)
        self.bar_volume = intraday["volume"].to_numpy(float)

    def price_at(self, ts: pd.Timestamp):
        """(price, when it became known) for the last price known at ts, or (None, None)."""
        i = np.searchsorted(self.bar_end, ts.value, side="right") - 1
        j = np.searchsorted(self.close_time, ts.value, side="right") - 1
        if i >= 0 and (j < 0 or self.bar_end[i] > self.close_time[j]):
            return self.bar_close[i], self.bar_end[i]
        if j >= 0:
            return self.daily_close[j], self.close_time[j]
        return None, None

    def volume_between(self, start: pd.Timestamp, end: pd.Timestamp) -> float:
        """Shares traded in the bars that fall entirely inside start..end."""
        a = np.searchsorted(self.bar_start, start.value, side="left")
        b = np.searchsorted(self.bar_end, end.value, side="right")
        return self.bar_volume[a:b].sum()

    def closes_before(self, ts: pd.Timestamp) -> int:
        """How many daily closes were known by ts."""
        return int(np.searchsorted(self.close_time, ts.value, side="right"))

    def day_index(self, day):
        """Index of this ticker's daily bar for `day`, or None if it didn't trade that day."""
        i = int(np.searchsorted(self.close_time, market_time.session_close(day).value, side="left"))
        return i if i < len(self.daily_dates) and self.daily_dates[i] == day else None

    def open_gap_pct(self, day) -> float:
        i = self.day_index(day)
        return pct(self.daily_close[i - 1], self.daily_open[i]) if i else np.nan

    def trend_pct(self, k: int, days: int) -> float:
        return pct(self.daily_close[k - 1 - days], self.daily_close[k - 1]) if k - 1 - days >= 0 else np.nan

    def volatility_pct(self, k: int) -> float:
        closes = self.daily_close[max(0, k - config.VOLATILITY_DAYS - 1):k]
        return np.std(np.diff(closes) / closes[:-1]) * 100 if len(closes) > 5 else np.nan

    def usual_volume(self, k: int, minutes: float) -> float:
        volumes = self.daily_volume[max(0, k - config.VOLATILITY_DAYS):k]
        return volumes.mean() * minutes / SESSION_MINUTES if len(volumes) else np.nan


def load_prices():
    daily = load_latest(config.RAW_PRICES_DIR, "prices_daily_*.parquet")
    intraday = load_latest(config.RAW_PRICES_DIR, "prices_intraday_*.parquet")
    if config.MARKET_TICKER not in set(daily["symbol"]):
        raise SystemExit(f"No daily bars for {config.MARKET_TICKER} (the trading calendar). Re-run collect/prices.py.")
    intraday_by_symbol = dict(tuple(intraday.groupby("symbol")))
    empty = intraday.iloc[:0]
    prices = {symbol: TickerPrices(bars, intraday_by_symbol.get(symbol, empty))
              for symbol, bars in daily.groupby("symbol")}
    trading_days = sorted(set(prices[config.MARKET_TICKER].daily_dates))
    return prices, trading_days


def flat_threshold(daily_volatility_pct: float, minutes: float) -> float:
    """The up/down cutoff for one row, in %: per stock if FLAT_THRESHOLD_SCALED (NaN without
    enough history), else the fixed FLAT_THRESHOLD_PCT."""
    if not config.FLAT_THRESHOLD_SCALED:
        return config.FLAT_THRESHOLD_PCT
    usual_move = daily_volatility_pct * np.sqrt(minutes / SESSION_MINUTES)
    return config.FLAT_THRESHOLD_FRACTION * usual_move


def story_repeats(news_items: list) -> set:
    """IDs of articles that repeat a story: another article of the same Event Registry event
    was published less than STORY_REPEAT_HOURS before them."""
    window = pd.Timedelta(hours=config.STORY_REPEAT_HOURS)
    first_seen, repeats = {}, set()
    for item in sorted(news_items, key=lambda a: a.get(config.NEWS_TIME_FIELD) or "9999"):
        event, when = item.get("eventUri"), item.get(config.NEWS_TIME_FIELD)
        if not (event and when):
            continue
        when = pd.to_datetime(when, utc=True)
        first = first_seen.setdefault(event, when)
        if first != when and when - first < window:
            repeats.add(item["uri"])
    return repeats


def build():
    if not config.TAGGED_NEWS_PATH.exists():
        raise SystemExit("No tagged news found. Run Jobs 1 and 2 first.")
    news_items = json.loads(config.TAGGED_NEWS_PATH.read_text())
    prices, trading_days = load_prices()
    day_set = set(trading_days)
    market = prices[config.MARKET_TICKER]
    pre_news = pd.Timedelta(minutes=config.PRE_NEWS_MINUTES)

    repeats = story_repeats(news_items)
    if config.DATASET_SKIP_STORY_REPEATS:
        news_items = [item for item in news_items if item["uri"] not in repeats]
        print(f"Skipped {len(repeats)} repeat reports of the same story (DATASET_SKIP_STORY_REPEATS)")

    rows = []
    used_articles = {}  # article_id -> item, for the articles that got at least one row
    skipped = Counter()
    fixed_cutoff_rows = 0

    for item in news_items:
        # One article can affect several companies (e.g. an automaker
        # roundup touching Ford, GM, and Tesla). Each is a separate training
        # example -- same text, different price outcome -- so explode into
        # one row per ticker rather than keeping only the first.
        # Plus a row for each market index fund (INDEX_TICKERS), even if Job 2 tagged nothing.
        tickers_field = market_time.tickers_to_check(item)
        if not tickers_field:
            skipped["article has no tickers"] += 1
            continue

        news_ts = market_time.news_time(item)
        if news_ts is None:
            skipped[f"published before {config.MIN_PUBLISH_DATE} (archive re-crawl)"] += 1
            continue
        if news_ts.date() < trading_days[0]:
            skipped["article is before the price data starts"] += 1
            continue
        entry_ts = market_time.entry_time(news_ts, trading_days)
        if entry_ts is None:
            skipped["article is after the price data ends"] += 1
            continue
        exit_ts = market_time.exit_for_entry(entry_ts)  # the close, if that comes first
        label_minutes = (exit_ts - entry_ts).total_seconds() / 60
        entry_day = market_time.market_date(entry_ts)
        entry_day_open = market_time.session_open(entry_day)
        at_open = entry_ts == entry_day_open  # news while the market was closed: trade the next open
        start_ts = news_ts - pre_news

        session = market_time.news_session(news_ts, day_set)
        local_news = news_ts.tz_convert(market_time.MARKET_TZ)
        source_name = (item.get("source") or {}).get("title", "")

        # Volume over the pre-news window, clipped to today's open (news at 9:45 -> 15 minutes)
        volume_from, volume_minutes = None, 0.0
        if session == "market_hours":
            volume_from = max(start_ts, market_time.session_open(local_news.date()))
            volume_minutes = (news_ts - volume_from).total_seconds() / 60

        spy_start, _ = market.price_at(start_ts)
        spy_news, _ = market.price_at(news_ts)
        spy_gap = market.open_gap_pct(entry_day)
        # The market's own move over the same window as the label, for market-relative
        # labels and hedged backtests (not a feature: it's in the future at entry time)
        spy_day = market.day_index(entry_day)
        spy_entry = (market.daily_open[spy_day] if spy_day is not None else None) if at_open else spy_news
        market_pct_change = pct(spy_entry, market.price_at(exit_ts)[0])

        roles = item.get("ticker_roles", {})
        for ticker in tickers_field:
            p = prices.get(ticker)
            if p is None:
                skipped["no prices for ticker"] += 1
                continue

            news_price, _ = p.price_at(news_ts)
            if at_open:
                day_idx = p.day_index(entry_day)
                entry_price = p.daily_open[day_idx] if day_idx is not None else None
            else:
                entry_price = news_price
            if entry_price is None:
                skipped["no price to enter at"] += 1
                continue
            exit_price, known_at = p.price_at(exit_ts)
            if exit_price is None or known_at <= entry_day_open.value:
                skipped["no trade between the open and the label's end"] += 1
                continue
            pct_change = pct(entry_price, exit_price)

            k = p.closes_before(news_ts)
            volatility = p.volatility_pct(k)
            threshold = flat_threshold(volatility, label_minutes)
            if threshold != threshold:  # NaN: under a week of price history
                threshold = config.FLAT_THRESHOLD_PCT
                fixed_cutoff_rows += 1
            if pct_change > threshold:
                label = "up"
            elif pct_change < -threshold:
                label = "down"
            else:
                label = "flat"

            start_price, _ = p.price_at(start_ts)
            volume_ratio = np.nan
            if volume_minutes >= config.INTRADAY_BAR_MINUTES:
                usual = p.usual_volume(k, volume_minutes)
                if usual == usual:  # not NaN
                    volume_ratio = np.log((p.volume_between(volume_from, news_ts) + 1) / (usual + 1))

            used_articles[item["uri"]] = item
            rows.append({
                "ticker": ticker,
                # how the news reaches this company; "index" = an INDEX_TICKERS row Job 2 didn't tag
                "role": roles.get(ticker, "index" if ticker in config.INDEX_TICKERS else "unknown"),
                # 1 = most affected, per the tagging model; 0 = not tagged (an index row)
                "ticker_rank": item["tickers"].index(ticker) + 1 if ticker in item.get("tickers", []) else 0,
                "n_tickers": len(item.get("tickers", [])),
                "news_timestamp": news_ts,
                "news_session": session,
                "day_of_week": local_news.strftime("%a"),
                "entry": "next_open" if at_open else "at_news",
                "hours_to_entry": (entry_ts - news_ts).total_seconds() / 3600,
                "label_minutes": label_minutes,
                "entry_timestamp": entry_ts,
                "exit_timestamp": exit_ts,
                "article_id": item["uri"],  # its text is in ARTICLES_PATH
                "source_name": source_name,
                "url": item.get("url", ""),
                "event_uri": item.get("eventUri"),
                "is_story_repeat": item["uri"] in repeats,  # another outlet reported it earlier that day
                "language": item.get("lang"),
                "sentiment_score": item.get("sentiment") if item.get("sentiment") is not None else 0.0,
                # Nested object whose exact shape isn't confirmed yet -- kept as JSON text.
                "location": json.dumps(item["location"]) if item.get("location") else None,
                # What the model sees (all known at entry_timestamp)
                "pre_news_pct": pct(start_price, news_price),
                "spy_pre_news_pct": pct(spy_start, spy_news),
                "open_gap_pct": p.open_gap_pct(entry_day),
                "spy_open_gap_pct": spy_gap,
                "volume_ratio": volume_ratio,
                **{f"ret_{days}d_pct": p.trend_pct(k, days) for days in config.TREND_DAYS},
                f"volatility_{config.VOLATILITY_DAYS}d_pct": volatility,
                "flat_threshold_pct": threshold,  # this row's up/down cutoff, not a feature
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pct_change": pct_change,  # for backtest P&L only, not a feature or label
                "market_pct_change": market_pct_change,  # MARKET_TICKER over the same window; not a feature
                "label": label,
            })

    dataset = pd.DataFrame(rows)
    print(f"Built {len(dataset)} labeled rows from {len(news_items)} news items")
    for reason, count in skipped.most_common():
        print(f"  skipped {count}: {reason}")
    if len(news_items) > 0:
        print(f"Average rows per article (multi-ticker explosion): {len(dataset) / len(news_items):.2f}")

    if dataset.empty:
        raise SystemExit("Dataset is empty -- check that tickers/date ranges line up between news and price data.")

    # --- Sanity checks you should actually look at, not skip past ---
    print("\nLabel distribution:")
    print(dataset["label"].value_counts(normalize=True))

    if config.FLAT_THRESHOLD_SCALED:
        cutoffs = dataset.groupby("ticker")["flat_threshold_pct"].median()
        print(f"\nPer-stock up/down cutoff (% move in {config.HORIZON_MINUTES} min), median across tickers: "
              f"{cutoffs.median():.3f}; 5th-95th percentile: {cutoffs.quantile(0.05):.3f} to {cutoffs.quantile(0.95):.3f}")
        for ticker in dataset["ticker"].value_counts().head(5).index:
            print(f"  {ticker:6s} {cutoffs[ticker]:.3f}")
        if fixed_cutoff_rows:
            print(f"  {fixed_cutoff_rows} rows used the fixed {config.FLAT_THRESHOLD_PCT}% (under a week of history)")

    print("\nWhen the news came out (share of rows):")
    print(dataset["news_session"].value_counts(normalize=True))

    print("\nHow the news reaches the company (share of rows):")
    print(dataset["role"].value_counts(normalize=True))

    print("\nPrice inputs and the label's move, in % (volume_ratio is missing for news")
    print("outside market hours, so expect about that share missing there):")
    numeric = dataset[["pre_news_pct", "spy_pre_news_pct", "open_gap_pct", "volume_ratio", "ret_1d_pct",
                       f"volatility_{config.VOLATILITY_DAYS}d_pct", "pct_change"]]
    summary = numeric.describe(percentiles=[0.05, 0.5, 0.95]).T[["5%", "50%", "95%"]]
    summary["missing"] = numeric.isna().mean()
    print(summary.round(3))

    print("\nTop 10 tickers by count:")
    print(dataset["ticker"].value_counts().head(10))

    print(f"\nDate range: {dataset['news_timestamp'].min()} to {dataset['news_timestamp'].max()}")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(DATASET_PATH, index=False)
    articles = pd.DataFrame({
        "article_id": list(used_articles),
        "headline": [a.get("title") or "" for a in used_articles.values()],
        "article_text": [a.get("body") or "" for a in used_articles.values()],
    })
    articles.to_parquet(ARTICLES_PATH, index=False)
    print(f"\nSaved to {DATASET_PATH}, and {len(articles)} articles' text to {ARTICLES_PATH}")


if __name__ == "__main__":
    build()
