"""
Pull historical prices from Alpaca for every ticker that Job 2 tagged, plus
config.MARKET_TICKER (SPY), so modeling/dataset.py can line news up with
prices. Two kinds of bars, both saved as parquet in data/raw_prices/:

  - daily bars (prices_daily_*.parquet), from a few weeks before the oldest
    tagged article: the closes the label exits at, the trend and
    volatility inputs, and the market's trading days (from SPY's).
  - INTRADAY_BAR_MINUTES bars (prices_intraday_*.parquet), regular session
    only, and only on the days the model needs them for some article: the
    day its label window ends (see market_time.py) and the day its pre-news
    window starts. Downloading every day for every ticker would take hours.

Daily bars come first, since the trading days decide which intraday days
are needed. Bars are split-adjusted, so a stock split doesn't look like a
crash. Tickers Alpaca has no prices for (wrong or non-US tickers from
Job 2) are listed at the end.

Reads data/processed/news_tagged.json (Job 2's output) -- tickers only exist
after AI tagging. Run it after job2-tag-collect, and again whenever new
articles have been tagged.

Run:
    python -m newsmodel.collect.prices
"""

import json
import re
import time
from collections import defaultdict
from datetime import date, datetime, timedelta

import pandas as pd

from newsmodel import config, market_time

SYMBOLS_PER_REQUEST = 200
# Alpaca's symbol format (e.g. BRK.B). Anything else (7203.T, BRK-B) is rejected by the
# API and would fail the whole request it's in, so it's left out up front.
VALID_SYMBOL = re.compile(r"^[A-Z]{1,5}(\.[A-Z]{1,2})?$")
BAR_COLUMNS = ["symbol", "timestamp", "open", "close", "volume"]


def get_unique_tickers(news_items):
    tickers = set()
    for item in news_items:
        tickers.update(item.get("tickers", []))
    return sorted(tickers)


def fetch_bars(client, symbols: list, timeframe, start: datetime, end: datetime) -> pd.DataFrame:
    """Bars for several symbols in one request (the SDK pages through them).
    Rate limits and server errors are retried with a growing wait."""
    from alpaca.common.exceptions import APIError
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest

    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=timeframe,
        start=start,
        end=end,
        adjustment=Adjustment.SPLIT,
        feed=DataFeed.SIP,  # all US exchanges; IEX alone sees ~2% of volume
    )
    for attempt in range(4):
        try:
            df = client.get_stock_bars(request).df
            break
        except APIError as e:
            retryable = e.status_code is None or e.status_code == 429 or e.status_code >= 500
            if not retryable or attempt == 3:
                raise
            print(f"  Alpaca error {e.status_code}, retrying in {60 * (attempt + 1)}s")
            time.sleep(60 * (attempt + 1))
    if df.empty:
        return pd.DataFrame(columns=BAR_COLUMNS)
    return df.reset_index()[BAR_COLUMNS]


def fetch_daily(client, tickers: list, start: datetime, end: datetime) -> pd.DataFrame:
    from alpaca.data.timeframe import TimeFrame

    frames = []
    for i in range(0, len(tickers), SYMBOLS_PER_REQUEST):
        chunk = tickers[i:i + SYMBOLS_PER_REQUEST]
        try:
            frames.append(fetch_bars(client, chunk, TimeFrame.Day, start, end))
        except Exception as e:
            print(f"  request failed ({e}); fetching its {len(chunk)} tickers one by one")
            for ticker in chunk:
                try:
                    frames.append(fetch_bars(client, [ticker], TimeFrame.Day, start, end))
                except Exception:
                    pass  # reported below as a ticker without prices
        print(f"  daily: {min(i + SYMBOLS_PER_REQUEST, len(tickers))}/{len(tickers)} tickers")
    return pd.concat(frames, ignore_index=True)


def intraday_days_needed(news_items: list, tickers_with_prices: set, trading_days: list) -> dict:
    """{trading day: tickers} -- the days each article's tickers need intraday bars."""
    day_set = set(trading_days)
    pre_news = pd.Timedelta(minutes=config.PRE_NEWS_MINUTES)
    needed = defaultdict(set)
    for item in news_items:
        tickers = [t for t in market_time.tickers_to_check(item) if t in tickers_with_prices]
        news_ts = market_time.news_time(item)
        if not tickers or news_ts is None:
            continue
        exit_ts = market_time.exit_time(news_ts, trading_days)
        if exit_ts is None:
            continue
        days = {market_time.market_date(exit_ts)}
        if market_time.is_in_session(news_ts - pre_news, day_set):
            days.add(market_time.market_date(news_ts - pre_news))
        for day in days:
            needed[day].update(tickers)
            needed[day].add(config.MARKET_TICKER)
    return needed


def fetch_intraday(client, needed: dict) -> pd.DataFrame:
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    timeframe = TimeFrame(config.INTRADAY_BAR_MINUTES, TimeFrameUnit.Minute)
    frames, failed_requests = [], 0
    for n, day in enumerate(sorted(needed), start=1):
        symbols = sorted(needed[day])
        start, end = market_time.session_open(day), market_time.session_close(day)
        for i in range(0, len(symbols), SYMBOLS_PER_REQUEST):
            try:
                bars = fetch_bars(client, symbols[i:i + SYMBOLS_PER_REQUEST], timeframe, start, end)
            except Exception as e:
                failed_requests += 1
                print(f"  FAILED {day} ({len(symbols[i:i + SYMBOLS_PER_REQUEST])} tickers): {e}")
                continue
            frames.append(bars[pd.to_datetime(bars["timestamp"], utc=True) < end])
        if n % 25 == 0 or n == len(needed):
            print(f"  intraday: {n}/{len(needed)} days")
    if failed_requests:
        print(f"{failed_requests} intraday requests failed; their articles will be skipped in the dataset")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=BAR_COLUMNS)


def main():
    if not (config.ALPACA_API_KEY and config.ALPACA_SECRET_KEY):
        raise SystemExit("Set ALPACA_API_KEY and ALPACA_SECRET_KEY as environment variables.")

    from alpaca.data.historical import StockHistoricalDataClient

    client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)

    if not config.TAGGED_NEWS_PATH.exists():
        raise SystemExit("No tagged news found. Run job1-fetch-news, then job2-tag-submit and job2-tag-collect first.")
    # Only articles the dataset can use: with something to check and a trustworthy news time
    news_items = [a for a in json.loads(config.TAGGED_NEWS_PATH.read_text())
                  if market_time.tickers_to_check(a) and market_time.news_time(a) is not None]
    all_tickers = sorted(set(get_unique_tickers(news_items)) | set(config.INDEX_TICKERS))
    tickers = sorted({t for t in all_tickers if VALID_SYMBOL.match(t)} | {config.MARKET_TICKER})
    malformed = [t for t in all_tickers if not VALID_SYMBOL.match(t)]

    oldest = min(market_time.news_time(a) for a in news_items).tz_localize(None).to_pydatetime()
    # Enough trading days before the oldest article for the volatility input (~1.5 calendar days each)
    start = oldest - timedelta(days=int(config.VOLATILITY_DAYS * 1.5) + 14)
    end = datetime.combine(date.today(), datetime.min.time())  # up to (not including) today

    print(f"Fetching daily bars for {len(tickers)} tickers from {start.date()}...")
    daily = fetch_daily(client, tickers, start, end)
    if config.MARKET_TICKER not in set(daily["symbol"]):
        raise SystemExit(f"No daily bars for {config.MARKET_TICKER}, so no trading calendar. Check the Alpaca keys.")
    spy_dates = pd.to_datetime(daily.loc[daily["symbol"] == config.MARKET_TICKER, "timestamp"], utc=True)
    trading_days = sorted(spy_dates.dt.tz_convert(market_time.MARKET_TZ).dt.date.unique())

    with_prices = set(daily["symbol"])
    needed = intraday_days_needed(news_items, with_prices, trading_days)
    print(f"\nFetching {config.INTRADAY_BAR_MINUTES}-minute bars for {len(needed)} trading days "
          f"({sum(len(s) for s in needed.values())} ticker-days)...")
    intraday = fetch_intraday(client, needed)

    config.RAW_PRICES_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for name, df in [("daily", daily), ("intraday", intraday)]:
        path = config.RAW_PRICES_DIR / f"prices_{name}_{stamp}.parquet"
        df.assign(timestamp=pd.to_datetime(df["timestamp"], utc=True)).to_parquet(path, index=False)
        print(f"Saved {len(df)} {name} bars to {path}")

    no_prices = [t for t in tickers if t not in with_prices]
    print(f"\n{len(with_prices)} tickers have prices. No prices for {len(no_prices)} "
          f"(wrong, delisted or non-US?): {no_prices[:20]}...")
    if malformed:
        print(f"Skipped {len(malformed)} tickers not in Alpaca's symbol format: {malformed[:20]}...")


if __name__ == "__main__":
    main()
