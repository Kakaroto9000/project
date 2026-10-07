"""
Combine tagged news + prices into a single labeled training table
(data/processed/labeled_dataset.parquet). First step of Job 3.

Reads data/processed/news_tagged.json (Job 2's output) and the latest
data/raw_prices/prices_*.json.

For each (article, ticker) pair:
  1. Entry price: the last daily close BEFORE the news came out. A daily
     close is treated as known at 4pm New York time on its date (Alpaca
     stamps the bar at the start of the day, before the close exists).
  2. Exit price: HORIZON_DAYS trading-day closes after the entry. With
     HORIZON_DAYS = 1 that's the first close after the news, so the label
     covers the market's reaction:
        Saturday news    -> Friday close  to Monday close
        Monday 8am/10am  -> Friday close  to Monday close
        Monday 6pm       -> Monday close  to Tuesday close
     Weekends and holidays have no bars, so they're skipped automatically.
  3. Label = "up" / "down" / "flat" -- always one of these three, never a
     percentage. FLAT_THRESHOLD_PCT decides where "flat" starts.
  4. news_session: when the article came out relative to the US market --
     "weekend", "holiday", "pre_market", "market_hours" or "after_hours".
     Fed to the model alongside the text, since news that waits a weekend
     to be priced in can behave differently from news during trading.
  5. role: how the news reaches this company, from Job 2 -- "direct",
     "competitor", "opposite_market", "supplier_customer", "world_event",
     "policy_macro" or "other" (the list is ROLES in collect/tickers.py).
     Fed to the model too: the same article can lift the company it's about
     and sink its competitor. ticker_rank (1 = most affected) is stored only.
  6. "headline" + "article_text" (Event Registry's full body) are the text
     the model trains on. "sentiment_score" (Event Registry's own -1..1
     float, 0.0 if missing) is stored for reference only.

Run (normally via job3-train):
    python -m newsmodel.modeling.dataset
"""

import json
from pathlib import Path

import pandas as pd

from newsmodel import config

DATASET_PATH = config.PROCESSED_DIR / "labeled_dataset.parquet"

MARKET_TZ = "America/New_York"
MARKET_OPEN = pd.Timedelta(hours=9, minutes=30)
MARKET_CLOSE = pd.Timedelta(hours=16)  # half-day sessions (1pm close) are ignored


def load_latest(directory: Path, pattern: str):
    files = sorted(directory.glob(pattern))
    if not files:
        raise SystemExit(f"No files matching {pattern} in {directory}. Run Jobs 1 and 2 first.")
    with open(files[-1]) as f:
        return json.load(f)


def prepare_bars(bars: list) -> pd.DataFrame:
    df = pd.DataFrame(bars)
    ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    timestamps = pd.to_datetime(df[ts_col], utc=True)
    trade_date = pd.to_datetime(timestamps.dt.date)
    df["trade_date"] = trade_date.dt.date
    df["close_time"] = (trade_date + MARKET_CLOSE).dt.tz_localize(MARKET_TZ).dt.tz_convert("UTC")
    return df.sort_values("close_time").reset_index(drop=True)


def find_entry_and_exit(df: pd.DataFrame, news_ts: pd.Timestamp, horizon: int):
    """Entry: last close at or before the news. Exit: `horizon` trading-day closes after that."""
    pos = df["close_time"].searchsorted(news_ts, side="right") - 1
    if pos < 0 or pos + horizon >= len(df):
        return None, None
    return df.iloc[pos], df.iloc[pos + horizon]


def news_session(news_ts: pd.Timestamp, trading_days: set) -> str:
    local = news_ts.tz_convert(MARKET_TZ)
    if local.date() not in trading_days:
        return "weekend" if local.weekday() >= 5 else "holiday"
    time_of_day = local - local.normalize()
    if time_of_day < MARKET_OPEN:
        return "pre_market"
    if time_of_day < MARKET_CLOSE:
        return "market_hours"
    return "after_hours"


def story_repeats(news_items: list) -> set:
    """IDs of articles that repeat a story: another article of the same Event Registry event
    was published less than STORY_REPEAT_HOURS before them."""
    window = pd.Timedelta(hours=config.STORY_REPEAT_HOURS)
    first_seen, repeats = {}, set()
    for item in sorted(news_items, key=lambda a: a.get("dateTime") or "9999"):
        event, when = item.get("eventUri"), item.get("dateTime")
        if not (event and when):
            continue
        when = pd.to_datetime(when, utc=True)
        first = first_seen.setdefault(event, when)
        if first != when and when - first < window:
            repeats.add(item["uri"])
    return repeats


def build():
    if config.HORIZON_DAYS < 1:
        raise SystemExit("HORIZON_DAYS must be at least 1 (1 = the first close after the news).")

    if not config.TAGGED_NEWS_PATH.exists():
        raise SystemExit("No tagged news found. Run Jobs 1 and 2 first.")
    news_items = json.loads(config.TAGGED_NEWS_PATH.read_text())
    raw_prices = load_latest(config.RAW_PRICES_DIR, "prices_*.json")

    price_dfs = {ticker: prepare_bars(bars) for ticker, bars in raw_prices.items() if bars}
    trading_days = set().union(*(set(df["trade_date"]) for df in price_dfs.values())) if price_dfs else set()

    repeats = story_repeats(news_items)
    if config.DATASET_SKIP_STORY_REPEATS:
        news_items = [item for item in news_items if item["uri"] not in repeats]
        print(f"Skipped {len(repeats)} repeat reports of the same story (DATASET_SKIP_STORY_REPEATS)")

    rows = []
    skipped_no_price = 0

    for item in news_items:
        # One article can affect several companies (e.g. an automaker
        # roundup touching Ford, GM, and Tesla). Each is a separate training
        # example -- same text, different price outcome -- so explode into
        # one row per ticker rather than keeping only the first.
        tickers_field = item.get("tickers", [])
        if config.TICKERS_UNIVERSE:
            tickers_field = [t for t in tickers_field if t in config.TICKERS_UNIVERSE]
        if not tickers_field:
            skipped_no_price += 1
            continue

        news_ts = pd.to_datetime(item.get("dateTime"), utc=True)
        session = news_session(news_ts, trading_days)
        source_name = (item.get("source") or {}).get("title", "")

        roles = item.get("ticker_roles", {})
        for ticker in tickers_field:
            if ticker not in price_dfs:
                skipped_no_price += 1
                continue

            entry_bar, exit_bar = find_entry_and_exit(price_dfs[ticker], news_ts, config.HORIZON_DAYS)
            if entry_bar is None:
                skipped_no_price += 1
                continue

            entry_price = entry_bar["close"]
            exit_price = exit_bar["close"]
            pct_change = (exit_price - entry_price) / entry_price * 100

            if pct_change > config.FLAT_THRESHOLD_PCT:
                label = "up"
            elif pct_change < -config.FLAT_THRESHOLD_PCT:
                label = "down"
            else:
                label = "flat"

            rows.append({
                "ticker": ticker,
                "role": roles.get(ticker, "unknown"),  # how the news reaches this company
                "ticker_rank": item["tickers"].index(ticker) + 1,  # 1 = most affected, per the tagging model
                "news_timestamp": news_ts,
                "news_session": session,
                "entry_date": entry_bar["trade_date"],
                "exit_date": exit_bar["trade_date"],
                "headline": item.get("title", ""),
                "article_text": item.get("body", ""),
                "source_name": source_name,
                "url": item.get("url", ""),
                "event_uri": item.get("eventUri"),
                "is_story_repeat": item["uri"] in repeats,  # another outlet reported it earlier that day
                "language": item.get("lang"),
                "sentiment_score": item.get("sentiment") if item.get("sentiment") is not None else 0.0,
                # Nested object whose exact shape isn't confirmed yet -- kept as JSON text.
                "location": json.dumps(item["location"]) if item.get("location") else None,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pct_change": pct_change,  # for backtest P&L only, not a feature or label
                "label": label,
            })

    dataset = pd.DataFrame(rows)
    print(f"Built {len(dataset)} labeled rows from {len(news_items)} news items")
    print(f"({skipped_no_price} ticker-article pairs skipped: no tickers or no matching price data)")
    if len(news_items) > 0:
        print(f"Average rows per article (multi-ticker explosion): {len(dataset) / len(news_items):.2f}")

    if dataset.empty:
        raise SystemExit("Dataset is empty -- check that tickers/date ranges line up between news and price data.")

    # --- Sanity checks you should actually look at, not skip past ---
    print("\nLabel distribution:")
    print(dataset["label"].value_counts(normalize=True))

    print("\nWhen the news came out (share of rows):")
    print(dataset["news_session"].value_counts(normalize=True))

    print("\nHow the news reaches the company (share of rows):")
    print(dataset["role"].value_counts(normalize=True))

    print("\nTop 10 tickers by count:")
    print(dataset["ticker"].value_counts().head(10))

    print(f"\nDate range: {dataset['news_timestamp'].min()} to {dataset['news_timestamp'].max()}")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(DATASET_PATH, index=False)
    print(f"\nSaved to {DATASET_PATH}")


if __name__ == "__main__":
    build()
