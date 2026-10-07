"""
Pull historical daily price bars from Alpaca for every ticker that Job 2
tagged, so modeling/dataset.py can align news timestamps to prices.

Reads data/processed/news_tagged.json (Job 2's output) -- tickers only exist
after AI tagging. Prices start a week before the
oldest tagged article, so even that one has a close before it. Run it
after job2-tag-collect, and again whenever new articles have been tagged.

Run:
    python -m newsmodel.collect.prices
"""

import json
from datetime import date, datetime, timedelta

from newsmodel import config


def get_unique_tickers(news_items):
    tickers = set()
    for item in news_items:
        tickers.update(item.get("tickers", []))
    return sorted(tickers)


def fetch_bars_for_ticker(client, ticker: str, start: datetime):
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    request = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Day,  # daily bars; switch to TimeFrame.Hour or
                                   # TimeFrame.Minute if you want a finer
                                   # horizon than HORIZON_DAYS=1 in config.py
        start=start,
        end=datetime.combine(date.today(), datetime.min.time()),  # up to (not including) today
    )
    bars = client.get_stock_bars(request)
    return bars.df.reset_index().to_dict(orient="records") if not bars.df.empty else []


def main():
    if not (config.ALPACA_API_KEY and config.ALPACA_SECRET_KEY):
        raise SystemExit("Set ALPACA_API_KEY and ALPACA_SECRET_KEY as environment variables.")

    from alpaca.data.historical import StockHistoricalDataClient

    client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)

    if not config.TAGGED_NEWS_PATH.exists():
        raise SystemExit("No tagged news found. Run job1-fetch-news, then job2-tag-submit and job2-tag-collect first.")
    news_items = json.loads(config.TAGGED_NEWS_PATH.read_text())
    tickers = get_unique_tickers(news_items)
    oldest = min(datetime.fromisoformat(a["dateTime"][:10]) for a in news_items)
    start = oldest - timedelta(days=7)
    print(f"Fetching price history for {len(tickers)} unique tickers from {start.date()}...")

    all_bars = {}
    failed = []
    for i, ticker in enumerate(tickers):
        try:
            bars = fetch_bars_for_ticker(client, ticker, start)
            all_bars[ticker] = bars
            if i % 25 == 0:
                print(f"  [{i}/{len(tickers)}] {ticker}: {len(bars)} bars")
        except Exception as e:
            failed.append(ticker)
            print(f"  FAILED {ticker}: {e}")

    config.RAW_PRICES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = config.RAW_PRICES_DIR / f"prices_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(all_bars, f, default=str)

    print(f"\nSaved price data for {len(all_bars)} tickers to {out_path}")
    if failed:
        print(f"Failed to fetch {len(failed)} tickers (delisted/invalid?): {failed[:20]}...")


if __name__ == "__main__":
    main()
