"""
The company universe: UNIVERSE_SIZE of the most traded US stocks and funds,
checked against EVERY article (not just the ones Claude tagged), so a model
per company can learn how that company reacts to any news.

  1. Pick the universe from the daily bars collect/prices.py saved: highest
     average daily dollar volume over the last year, at least $5 a share,
     with prices on 80%+ of trading days, leaving out leveraged, inverse and
     other non-plain products by name (Alpaca's asset list). The most traded
     names are the cheapest to trade. Saved to data/raw_prices/universe.json.
  2. Download 1-minute bars for all of them on every trading day some article
     needs (the same days as collect/prices.py), one parquet file per day in
     data/raw_prices/universe_minute/. Days already saved are skipped, so a
     re-run after an interruption carries on where it stopped.

Keeps Windows awake while it runs. Needs ALPACA_API_KEY and ALPACA_SECRET_KEY.

Run (after collect/prices.py):
    python -m newsmodel.collect.universe              # regular session, 9:30-16:00
    python -m newsmodel.collect.universe --extended   # pre-market 4:00-9:30 and after hours 16:00-20:00
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from newsmodel import config, market_time
from newsmodel.collect import prices
from newsmodel.keep_awake import keep_awake

UNIVERSE_PATH = config.RAW_PRICES_DIR / "universe.json"
MINUTE_DIR = config.RAW_PRICES_DIR / "universe_minute"
EXT_MINUTE_DIR = config.RAW_PRICES_DIR / "universe_minute_ext"  # 4:00-9:30 and 16:00-20:00 New York time
EXT_SESSIONS = [(pd.Timedelta(hours=4), market_time.MARKET_OPEN), (market_time.MARKET_CLOSE, pd.Timedelta(hours=20))]
# Products that only repeat another ticker with leverage or a flipped sign, or aren't shares
NOT_PLAIN = re.compile(r"\b(?:2x|3x|-2x|-3x|ultra|ultrapro|bull|bear|leveraged|inverse|short|warrant|rights?|units?)\b", re.I)


def latest_daily() -> pd.DataFrame:
    files = sorted(config.RAW_PRICES_DIR.glob("prices_daily_*.parquet"))
    if not files:
        raise SystemExit("No daily bars yet. Run python -m newsmodel.collect.prices first.")
    daily = pd.read_parquet(files[-1])
    daily["date"] = pd.to_datetime(daily["timestamp"], utc=True).dt.tz_convert(market_time.MARKET_TZ).dt.date
    return daily


def asset_names() -> dict:
    """{symbol: name} for every US equity Alpaca lists."""
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import AssetClass
    from alpaca.trading.requests import GetAssetsRequest

    client = TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, paper=True)
    assets = client.get_all_assets(GetAssetsRequest(asset_class=AssetClass.US_EQUITY))
    return {a.symbol: a.name or "" for a in assets}


def pick_universe(daily: pd.DataFrame) -> list:
    days = sorted(daily.loc[daily["symbol"] == config.MARKET_TICKER, "date"].unique())
    recent = daily[daily["date"].isin(set(days[-250:]))]
    stats = pd.DataFrame({
        "days": daily.groupby("symbol")["date"].nunique(),
        "dollar_volume": (recent["close"] * recent["volume"]).groupby(recent["symbol"]).mean(),
        "price": recent.groupby("symbol")["close"].median(),
    }).dropna()
    stats = stats[(stats["days"] >= 0.8 * len(days)) & (stats["price"] >= 5)]
    names = asset_names()
    stats["name"] = stats.index.map(lambda s: names.get(s, ""))
    plain = stats[~stats["name"].str.contains(NOT_PLAIN)]
    print(f"Left out {len(stats) - len(plain)} leveraged/inverse/other products, e.g. "
          f"{list(stats[stats['name'].str.contains(NOT_PLAIN)].sort_values('dollar_volume', ascending=False).index[:8])}")
    top = plain.sort_values("dollar_volume", ascending=False).head(config.UNIVERSE_SIZE)
    universe = list(top.index)
    for fund in config.INDEX_TICKERS:  # the market/sector funds are always in
        if fund not in universe:
            universe.append(fund)
    last = top.iloc[-1]
    print(f"Universe: {len(universe)} tickers; smallest {top.index[-1]} trades ${last['dollar_volume'] / 1e6:,.0f}M a day")
    UNIVERSE_PATH.write_text(json.dumps([{"ticker": t, "name": names.get(t, "")} for t in universe], indent=1))
    return universe


def needed_days(trading_days: list) -> list:
    """Trading days some article's label or pre-news window falls on (as collect/prices.py)."""
    items = [a for a in json.loads(config.TAGGED_NEWS_PATH.read_text()) if market_time.news_time(a) is not None]
    return sorted(prices.intraday_days_needed(items, {config.MARKET_TICKER}, trading_days))


def needed_ext_days(trading_days: list) -> list:
    """Trading days an article's extended-hours window or pre-news hour can fall on: its own
    day if published 4:00-20:00 on a trading day, else the next trading day (from 4:00)."""
    import bisect
    days = set()
    for a in json.loads(config.TAGGED_NEWS_PATH.read_text()):
        t = market_time.news_time(a)
        if t is None:
            continue
        for ts in (t, t - pd.Timedelta(minutes=config.PRE_NEWS_MINUTES)):
            local = ts.tz_convert(market_time.MARKET_TZ)
            d, mins = local.date(), local.hour * 60 + local.minute
            if d in trading_days_set(trading_days) and 4 * 60 <= mins < 20 * 60:
                days.add(d)
            else:
                i = bisect.bisect_right(trading_days, d) if mins >= 20 * 60 else bisect.bisect_left(trading_days, d)
                if i < len(trading_days):
                    days.add(trading_days[i])
    return sorted(days)


_DAYSET = {}


def trading_days_set(trading_days):
    key = id(trading_days)
    if key not in _DAYSET:
        _DAYSET[key] = set(trading_days)
    return _DAYSET[key]


def fetch_day(client, day, universe: list, extended: bool = False) -> str:
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    path = (EXT_MINUTE_DIR if extended else MINUTE_DIR) / f"{day}.parquet"
    if path.exists():
        return "skip"
    timeframe = TimeFrame(config.INTRADAY_BAR_MINUTES, TimeFrameUnit.Minute)
    if extended:
        local = lambda off: (pd.Timestamp(day) + off).tz_localize(market_time.MARKET_TZ).tz_convert("UTC")
        windows = [(local(a), local(b)) for a, b in EXT_SESSIONS]
    else:
        windows = [(market_time.session_open(day), market_time.session_close(day))]
    frames = []
    for start, end in windows:
        for i in range(0, len(universe), prices.SYMBOLS_PER_REQUEST):
            bars = prices.fetch_bars(client, universe[i:i + prices.SYMBOLS_PER_REQUEST], timeframe, start, end)
            ts = pd.to_datetime(bars["timestamp"], utc=True)
            frames.append(bars[(ts >= start) & (ts < end)])
    df = pd.concat(frames, ignore_index=True)
    df = pd.DataFrame({
        "symbol": df["symbol"].astype("category"),
        "timestamp": pd.to_datetime(df["timestamp"], utc=True),
        "close": df["close"].astype("float32"),
        "volume": df["volume"].astype("float32"),
    })
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)  # a half-written day is never mistaken for a finished one
    return "done"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--extended", action="store_true", help="pre-market and after-hours bars (universe_minute_ext/)")
    args = ap.parse_args()
    if not (config.ALPACA_API_KEY and config.ALPACA_SECRET_KEY):
        raise SystemExit("Set ALPACA_API_KEY and ALPACA_SECRET_KEY as environment variables.")
    from alpaca.data.historical import StockHistoricalDataClient

    with keep_awake():
        daily = latest_daily()
        if UNIVERSE_PATH.exists():  # keep the same stocks (and order) the saved grids use
            universe = [u["ticker"] for u in json.loads(UNIVERSE_PATH.read_text())]
            print(f"Using the saved universe: {len(universe)} tickers", flush=True)
        else:
            universe = pick_universe(daily)
        trading_days = sorted(daily.loc[daily["symbol"] == config.MARKET_TICKER, "date"].unique())
        out_dir = EXT_MINUTE_DIR if args.extended else MINUTE_DIR
        days = needed_ext_days(trading_days) if args.extended else needed_days(trading_days)
        out_dir.mkdir(parents=True, exist_ok=True)
        todo = [d for d in days if not (out_dir / f"{d}.parquet").exists()]
        print(f"{len(days)} trading days needed, {len(days) - len(todo)} already saved, {len(todo)} to fetch "
              f"({config.INTRADAY_BAR_MINUTES}-minute bars, {len(universe)} tickers each)", flush=True)

        client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
        failed = []
        with ThreadPoolExecutor(max_workers=config.UNIVERSE_DOWNLOAD_THREADS) as pool:
            futures = {pool.submit(fetch_day, client, d, universe, args.extended): d for d in todo}
            for n, (future, day) in enumerate(futures.items(), start=1):
                try:
                    future.result()
                except Exception as e:
                    failed.append(day)
                    print(f"  FAILED {day}: {e}", flush=True)
                if n % 10 == 0 or n == len(todo):
                    print(f"  universe minute bars: {n}/{len(todo)} days", flush=True)
        if failed:
            print(f"{len(failed)} days failed; re-run to fetch just those: {failed[:10]}")
        else:
            print(f"All {len(days)} days saved in {out_dir}")


if __name__ == "__main__":
    main()
