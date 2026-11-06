"""
Delayed stock prices from Alpaca's free plan, and for one article: the model's
inputs (as training saw them) and what then actually happened in its window.

Everything is computed the way modeling/grid_dataset.py built the regular-hours
training grid (data/processed/grid) -- the per-stock numbers literally come from
grid_dataset.stock_columns, fed the same kind of bars training was fed:

  inputs    per stock    PRICE_NAMES: last hour's move, opening jump, volume vs.
                         usual, 1- and 5-day return, usual daily move
            per article  ARTICLE_NAMES: SPY's last hour and opening jump, hours
                         the news waited, window length, traded at the news or
                         the next open, news session
  outcome   entry price  the price at the news if the market was open, else the
                         entry day's official open (its daily bar)
            exit price   the last 1-minute bar's close by the window's end, which
                         must be a trade made on the entry day after 9:30
            move         % entry -> exit, minus SPY's for companies, own move for
                         INDEX_TICKERS funds; up/flat/down cut at the stock's
                         usual move over the window, like grid's y_main

Prices: all-exchange (SIP) bars, split-adjusted, regular session 9:30-16:00 only
-- what training used. The free plan refuses SIP data from the last
LIVE_DATA_DELAY_MINUTES, so an article is only processed once its window's end is
that old (ready_at). Daily bars and finished days' minute bars are cached, today's
minute bars are topped up as time goes on, so the next article costs a request or
two. On the entry day itself the official open comes from Alpaca's still-running daily
bar, so it's fetched again on every call until the day is over. Requests go out in batches of SYMBOLS_PER_REQUEST symbols and stay under the
free plan's 200 requests a minute.

Check it against the training grid:
    python -m newsmodel.live.check_market
"""

import datetime as dt
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from newsmodel import config, market_time
from newsmodel.collect.prices import BAR_COLUMNS, SYMBOLS_PER_REQUEST, fetch_bars
from newsmodel.modeling.grid_dataset import stock_columns
from newsmodel.modeling.universe_dataset import MIN, SESSION_MINUTES, price_at, rolling_daily

DAILY_HISTORY_DAYS = 50    # calendar days of daily bars before the entry day: 20-day volatility needs 21 closes
REQUESTS_PER_MINUTE = 180  # the free plan allows 200 (every page of a paged answer counts)
CACHED_DAYS = 6            # trading days of bars kept in memory
NO_BARS = {"start": np.array([], np.int64), "end": np.array([], np.int64), "close": np.array([]),
           "close64": np.array([]), "cumvol": np.array([0.0])}


class NotReady(Exception):
    """The prices needed are still inside the free plan's delay."""


def ns(ts) -> int:
    return pd.Timestamp(ts).tz_convert("UTC").tz_localize(None).value


def midnight(day) -> pd.Timestamp:
    return market_time._at(day, pd.Timedelta(0))


def local_date(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(market_time.MARKET_TZ).dt.date


_TRIM_LOCK = threading.Lock()


def _trim(cache: dict):
    with _TRIM_LOCK:
        while len(cache) > CACHED_DAYS:
            cache.pop(min(cache))


class Market:
    def __init__(self):
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.trading.client import TradingClient
        self.data = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
        self.trading = TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, paper=True)
        self._days, self._days_on = None, None
        self._hist = {}     # entry day -> (symbols, daily bars of the days before it)
        self._dd = {}       # entry day -> (symbols, final?, {symbol: rolling_daily numbers})
        self._minutes = {}  # trading day -> {"syms", "until", "df"}: that day's minute bars before "until"
        self._calls, self._lock = deque(), threading.Lock()  # request times in the last minute
        self._pool = ThreadPoolExecutor(8)   # requests of one fetch side by side
        self._tasks = ThreadPoolExecutor(4)  # daily and minute fetches of one article side by side

    # --- Calendar ---
    def trading_days(self) -> list:
        """Trading days from the start of 2025 to two weeks ahead (Alpaca's calendar), fetched once a day."""
        today = dt.date.today()
        if self._days_on != today:
            from alpaca.trading.requests import GetCalendarRequest
            cal = self._retry(lambda: self.trading.get_calendar(
                GetCalendarRequest(start=dt.date(2025, 1, 1), end=today + dt.timedelta(days=14))))
            self._days, self._days_on = sorted(c.date for c in cal), today
        return self._days

    def window(self, news_ts: pd.Timestamp):
        """(entry, exit) of the article's training window (market_time.py), or (None, None)."""
        entry = market_time.entry_time(news_ts, self.trading_days())
        return (None, None) if entry is None else (entry, market_time.exit_for_entry(entry))

    def ready_at(self, news_ts: pd.Timestamp):
        """When the whole window is old enough for the free plan's delayed prices."""
        _, exit_ts = self.window(news_ts)
        if exit_ts is None:
            return None
        return exit_ts + pd.Timedelta(minutes=config.LIVE_DATA_DELAY_MINUTES + config.LIVE_DATA_MARGIN_MINUTES)

    @staticmethod
    def data_until() -> pd.Timestamp:
        """SIP data is allowed up to here (a little inside the plan's delay, on a whole minute)."""
        return (pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=config.LIVE_DATA_DELAY_MINUTES, seconds=30)).floor("min")

    # --- Requests ---
    def _retry(self, call, attempts=4):
        """Network hiccups are retried; fetch_bars itself already waits out rate limits and server errors."""
        import requests
        for attempt in range(attempts):
            try:
                return call()
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if attempt == attempts - 1:
                    raise
                print(f"  Alpaca connection problem ({type(e).__name__}), retrying in {5 * (attempt + 1)}s")
                time.sleep(5 * (attempt + 1))

    def _throttle(self):
        """Wait until one more request fits in REQUESTS_PER_MINUTE."""
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= 60:
                    self._calls.popleft()
                if len(self._calls) < REQUESTS_PER_MINUTE:
                    self._calls.append(now)
                    return
                wait = 60 - (now - self._calls[0])
            time.sleep(wait)

    def _fetch(self, symbols, timeframe, start, end) -> pd.DataFrame:
        self._throttle()
        df = self._retry(lambda: fetch_bars(self.data, symbols, timeframe, start, end, "sip"))
        with self._lock:  # the SDK pages through 10,000 bars per request: count the extra pages too
            self._calls.extend([time.monotonic()] * (len(df) // 10_000))
        return df

    def _bars(self, symbols, timeframe, start, end) -> pd.DataFrame:
        """SIP bars for any number of symbols, in equal requests of at most SYMBOLS_PER_REQUEST
        symbols run side by side (207 symbols -> 2 x 104, not 200 + 7)."""
        if end > self.data_until() + pd.Timedelta(minutes=1):
            raise NotReady(f"prices up to {end} are still inside the free plan's {config.LIVE_DATA_DELAY_MINUTES}-minute delay")
        n = -(-len(symbols) // SYMBOLS_PER_REQUEST)
        chunks = [list(c) for c in np.array_split(np.array(symbols, dtype=object), n)]
        frames = [f for f in self._pool.map(lambda c: self._fetch(c, timeframe, start, end), chunks) if len(f)]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=BAR_COLUMNS)

    # --- Daily bars ---
    def daily(self, symbols: list, day) -> dict:
        """{symbol: rolling_daily numbers} over the ~DAILY_HISTORY_DAYS before `day` plus `day`'s own
        bar (its official open; its close isn't used by anything known before the news)."""
        from alpaca.data.timeframe import TimeFrame
        hit = self._dd.get(day)
        if hit and hit[1] and set(symbols) <= hit[0]:
            return hit[2]
        hist = self._hist.get(day)
        if not hist or not set(symbols) <= hist[0]:
            start = midnight(day) - pd.Timedelta(days=DAILY_HISTORY_DAYS)
            hist = (set(symbols), self._bars(symbols, TimeFrame.Day, start, midnight(day) - pd.Timedelta(minutes=1)))
            self._hist[day] = hist
            _trim(self._hist)
        # The day's own bar: complete once the next midnight is outside the delay; until then
        # it's fetched again each time (its open is all that's used, and stocks open one by one)
        day_end = midnight(day + dt.timedelta(days=1)) - pd.Timedelta(minutes=1)
        until = self.data_until()
        today = self._bars(symbols, TimeFrame.Day, midnight(day), min(day_end, until)) if until > midnight(day) else None
        bars = pd.concat([hist[1], today], ignore_index=True) if today is not None and len(today) else hist[1]
        bars = bars.assign(date=local_date(bars))
        bars = bars[bars["date"] <= day]
        dd = {sym: rolling_daily(d) for sym, d in bars.groupby("symbol")}
        self._dd[day] = (set(symbols), until >= day_end, dd)
        _trim(self._dd)
        return dd

    # --- Minute bars ---
    def _minute_day(self, symbols: list, day, until: pd.Timestamp) -> pd.DataFrame:
        """`day`'s regular-session 1-minute bars that start before `until`, topped up from the cache."""
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        lo, hi = market_time.session_open(day), min(market_time.session_close(day), until)
        c = self._minutes.get(day)
        if c is None or not set(symbols) <= c["syms"]:
            c = {"syms": set(symbols) | (c["syms"] if c else set()), "until": lo, "df": pd.DataFrame(columns=BAR_COLUMNS)}
        if c["until"] < hi:
            new = self._bars(sorted(c["syms"]), TimeFrame(config.INTRADAY_BAR_MINUTES, TimeFrameUnit.Minute), c["until"], hi)
            ts = pd.to_datetime(new["timestamp"], utc=True)
            new = new[(ts >= c["until"]) & (ts < hi)]  # the request's end is inclusive
            c = {**c, "until": hi, "df": pd.concat([c["df"], new], ignore_index=True) if len(c["df"]) else new}
            self._minutes[day] = c
            _trim(self._minutes)
        df = c["df"]
        return df[pd.to_datetime(df["timestamp"], utc=True) < until] if until < c["until"] else df

    def minutes(self, symbols: list, spans: dict) -> dict:
        """{symbol: bars} for the given trading days, as grid_dataset feeds stock_columns. Closes and
        volumes are rounded to float32 like the saved training bars; close64 keeps full precision
        (SPY's article inputs came from collect/prices.py's float64 bars)."""
        frames = list(self._tasks.map(lambda d: self._minute_day(symbols, d, spans[d]), spans))
        frames = [f for f in frames if len(f)]
        if not frames:
            return {}
        df = pd.concat(frames, ignore_index=True)
        df = df.assign(timestamp=pd.to_datetime(df["timestamp"], utc=True)).sort_values(["symbol", "timestamp"])
        out = {}
        for sym, m in df[df["symbol"].isin(symbols)].groupby("symbol"):
            start = m["timestamp"].dt.tz_localize(None).to_numpy("datetime64[ns]").astype(np.int64)
            close64 = m["close"].to_numpy(np.float64)
            out[sym] = {"start": start, "end": start + config.INTRADAY_BAR_MINUTES * MIN,
                        "close": close64.astype(np.float32).astype(np.float64), "close64": close64,
                        "cumvol": np.concatenate([[0], np.cumsum(m["volume"].to_numpy(np.float32).astype(np.float64))])}
        return out

    # --- One article ---
    def _compute(self, symbols: list, news_ts: pd.Timestamp, entry_ts: pd.Timestamp, exit_ts: pd.Timestamp, need: pd.Timestamp):
        """Everything stock_columns gives for one article, from the prices known up to `need`."""
        if need > self.data_until():
            raise NotReady(f"prices up to {need} are still inside the free plan's {config.LIVE_DATA_DELAY_MINUTES}-minute delay")
        days = self.trading_days()
        syms = list(dict.fromkeys(list(symbols) + [config.MARKET_TICKER]))  # SPY: the article inputs and the hedge
        news_day, entry_day = market_time.market_date(news_ts), market_time.market_date(entry_ts)
        entry_open = market_time.session_open(entry_day)
        session = market_time.news_session(news_ts, set(days))
        # Minute bars needed: the entry day's session up to the exit, and for news on an earlier day
        # its session up to the news, unless the whole last hour was after the close. Earlier bars
        # never matter: at a time before a day's open price_at finds the previous daily close
        # (it prefers that over the 15:59 bar, which ends at the same moment).
        spans = {entry_day: min(exit_ts, self.data_until())}
        news_open, news_close = market_time.session_open(news_day), market_time.session_close(news_day)
        if news_day != entry_day and news_day in set(days) and news_ts > news_open and \
                news_ts - pd.Timedelta(minutes=config.PRE_NEWS_MINUTES) < news_close:
            spans[news_day] = min(news_ts, news_close)
        daily = self._tasks.submit(self.daily, syms, entry_day)
        bars = self.minutes(syms, spans)
        daily = daily.result()

        art = pd.DataFrame({"news_ns": [ns(news_ts)], "entry_ns": [ns(entry_ts)], "exit_ns": [ns(exit_ts)],
                            "entry_open_ns": [ns(entry_open)], "news_day_open_ns": [ns(news_open)],
                            "at_news": [entry_ts != entry_open], "market_hours": [session == "market_hours"]})
        n = len(syms)
        prices, raw = np.full((n, 6), np.nan), np.full(n, np.nan)
        entry_p, exit_p = np.full(n, np.nan), np.full(n, np.nan)
        news, exit_, eo = ns(news_ts), ns(exit_ts), ns(entry_open)
        for i, sym in enumerate(syms):
            dd = daily.get(sym)
            if dd is None:  # training skipped stocks without daily bars too
                continue
            b = bars.get(sym, NO_BARS)
            with np.errstate(invalid="ignore", divide="ignore"):  # no bars yet: NaN, as in training
                main, _, p = stock_columns(art, dd, b)
            raw[i], prices[i] = main[0], p[0]
            # The same entry and exit stock_columns divides, kept for the pretend trade
            ct = dd["close_time"]
            ke = min(np.searchsorted(ct, eo, side="left"), len(ct) - 1)
            day_ok = len(ct) and ct[ke] > eo and ct[ke] - eo < 24 * 60 * MIN
            if day_ok:
                entry_p[i] = price_at(np.array([news]), b["end"], b["close"], ct, dd["close"])[0][0] if entry_ts != entry_open else dd["open"][ke]
            j = np.searchsorted(b["end"], exit_, side="right") - 1
            if day_ok and j >= 0 and b["end"][j] > eo:
                exit_p[i] = b["close"][j]
        spy = syms.index(config.MARKET_TICKER)
        sd, sb = daily.get(config.MARKET_TICKER), bars.get(config.MARKET_TICKER, NO_BARS)
        if sd is not None:  # SPY's last hour from full-precision bars, as modeling/dataset.py did
            p0, p1 = (price_at(np.array([t]), sb["end"], sb["close64"], sd["close_time"], sd["close"])[0][0]
                      for t in (news - config.PRE_NEWS_MINUTES * MIN, news))
            spy_pre = (p1 - p0) / p0 * 100
        else:
            spy_pre = np.nan
        article = np.array([spy_pre, prices[spy, 1], (entry_ts - news_ts).total_seconds() / 3600,
                            (exit_ts - entry_ts).total_seconds() / 60, entry_ts != entry_open,
                            session == "pre_market", session == "market_hours", session == "after_hours",
                            session in ("weekend", "holiday")], np.float32)
        keep = slice(0, len(symbols))  # drop SPY again if it was only added here
        return {"prices": prices[keep], "article": article, "raw": raw[keep], "spy_raw": raw[spy],
                "entry": entry_p[keep], "exit": exit_p[keep], "vol20": prices[keep, 5]}

    def inputs(self, symbols: list, news_ts: pd.Timestamp, entry_ts: pd.Timestamp, exit_ts: pd.Timestamp):
        """(prices (len(symbols), 6) in grid_dataset.PRICE_NAMES order, article (9,) in ARTICLE_NAMES
        order); NaN where unknown. Needs the entry day's open, so call it once the entry is past the delay."""
        need = max(news_ts, entry_ts + pd.Timedelta(minutes=1))
        c = self._compute(symbols, news_ts, entry_ts, exit_ts, need)
        return c["prices"], c["article"]

    def outcome(self, symbols: list, news_ts: pd.Timestamp, entry_ts: pd.Timestamp, exit_ts: pd.Timestamp) -> dict:
        """What happened in the window, per stock: entry_price, exit_price, raw_move (%), move (% vs.
        SPY for companies, own move for INDEX_TICKERS), threshold (%) and label "up"/"flat"/"down"
        (None without both prices). Call it once the exit is past the delay (ready_at)."""
        c = self._compute(symbols, news_ts, entry_ts, exit_ts, exit_ts)
        # float32 like grid_dataset, so moves right at the cutoff get the same label
        raw = c["raw"].astype(np.float32)
        funds = np.isin(symbols, config.INDEX_TICKERS)
        move = np.where(funds, raw, raw - np.float32(c["spy_raw"]))
        minutes = (exit_ts - entry_ts).total_seconds() / 60
        usual = c["vol20"].astype(np.float32) * np.sqrt(minutes / SESSION_MINUTES)
        thr = np.where(np.isfinite(usual), config.FLAT_THRESHOLD_FRACTION * usual, config.FLAT_THRESHOLD_PCT)
        label = np.where(move > thr, "up", np.where(move < -thr, "down", "flat")).astype(object)
        label[~np.isfinite(move)] = None
        return {"entry_price": c["entry"], "exit_price": c["exit"], "raw_move": c["raw"],
                "move": move.astype(np.float64), "label": label, "threshold": thr}
