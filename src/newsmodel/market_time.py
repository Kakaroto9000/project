"""
The US market clock, shared by collect/prices.py and modeling/dataset.py:
when a piece of news came out relative to the market, and when the move the
model predicts starts and ends.

The label is the HORIZON_MINUTES move from the first moment you could trade
on the news: right away if the market is open, otherwise at the next open.
News late in the session is measured to the 16:00 close instead, as long as
at least LATE_NEWS_MIN_MINUTES are left -- big news often drops just before
the close:
    Monday 11:00 news   -> Monday 11:00 to 11:30
    Monday 15:45 news   -> Monday 15:45 to the 16:00 close (15 minutes)
    Monday 15:55 news   -> Tuesday 9:30 to 10:00 (too little time left)
    Monday 6pm news     -> Tuesday 9:30 to 10:00
    Saturday news       -> Monday 9:30 to 10:00
    Monday 8am news     -> Monday 9:30 to 10:00

Trading days come from the market ticker's daily bars (config.MARKET_TICKER),
so weekends and holidays are skipped automatically.
"""

import bisect

import pandas as pd

from newsmodel import config

MARKET_TZ = "America/New_York"
MARKET_OPEN = pd.Timedelta(hours=9, minutes=30)
MARKET_CLOSE = pd.Timedelta(hours=16)  # half-day sessions (1pm close) are ignored


def _at(day, offset: pd.Timedelta) -> pd.Timestamp:
    return (pd.Timestamp(day) + offset).tz_localize(MARKET_TZ).tz_convert("UTC")


def session_open(day) -> pd.Timestamp:
    return _at(day, MARKET_OPEN)


def session_close(day) -> pd.Timestamp:
    return _at(day, MARKET_CLOSE)


def news_time(item: dict):
    """When the news came out (config.NEWS_TIME_FIELD), or None if it's missing or
    before MIN_PUBLISH_DATE."""
    when = item.get(config.NEWS_TIME_FIELD)
    if not when:
        return None
    when = pd.to_datetime(when, utc=True)
    return when if when >= pd.Timestamp(config.MIN_PUBLISH_DATE, tz="UTC") else None


def tickers_to_check(item: dict) -> list:
    """The tickers one article gets a row for: Job 2's tags (narrowed to TICKERS_UNIVERSE
    if set), then each of INDEX_TICKERS not already among them."""
    tagged = item.get("tickers") or []
    if config.TICKERS_UNIVERSE:
        tagged = [t for t in tagged if t in config.TICKERS_UNIVERSE]
    return tagged + [t for t in config.INDEX_TICKERS if t not in tagged]


def market_date(ts: pd.Timestamp):
    """The New York calendar date of a UTC timestamp."""
    return ts.tz_convert(MARKET_TZ).date()


def is_in_session(ts: pd.Timestamp, trading_days: set) -> bool:
    day = market_date(ts)
    return day in trading_days and session_open(day) <= ts < session_close(day)


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


def entry_time(news_ts: pd.Timestamp, trading_days_sorted: list):
    """
    When the predicted move starts: at the news if the market is open and
    stays open LATE_NEWS_MIN_MINUTES longer, otherwise at the next open.
    None if the next trading day is past the end of the price data.
    """
    min_left = pd.Timedelta(minutes=config.LATE_NEWS_MIN_MINUTES)
    day = market_date(news_ts)
    i = bisect.bisect_left(trading_days_sorted, day)
    if i < len(trading_days_sorted) and trading_days_sorted[i] == day:
        if news_ts < session_open(day):
            return session_open(day)
        if news_ts + min_left <= session_close(day):
            return news_ts
        i += 1  # too late in the day: wait for tomorrow's open
    if i >= len(trading_days_sorted):
        return None
    return session_open(trading_days_sorted[i])


def exit_for_entry(entry: pd.Timestamp) -> pd.Timestamp:
    """When the predicted move ends: HORIZON_MINUTES after it starts, or at the close if that's sooner."""
    return min(entry + pd.Timedelta(minutes=config.HORIZON_MINUTES), session_close(market_date(entry)))


def exit_time(news_ts: pd.Timestamp, trading_days_sorted: list):
    entry = entry_time(news_ts, trading_days_sorted)
    return None if entry is None else exit_for_entry(entry)


# --- Extended hours: pre-market 4:00-9:30 and after hours 16:00-20:00 also trade ---
EXT_OPEN = pd.Timedelta(hours=4)
EXT_CLOSE = pd.Timedelta(hours=20)


def ext_entry_time(news_ts: pd.Timestamp, trading_days_sorted: list):
    """With extended hours: at the news if it's 4:00-20:00 on a trading day and at least
    LATE_NEWS_MIN_MINUTES are left, otherwise 4:00 the next trading day. Earnings at 16:05
    are traded at 16:05, not at the next morning's open."""
    min_left = pd.Timedelta(minutes=config.LATE_NEWS_MIN_MINUTES)
    day = market_date(news_ts)
    i = bisect.bisect_left(trading_days_sorted, day)
    if i < len(trading_days_sorted) and trading_days_sorted[i] == day:
        if news_ts < _at(day, EXT_OPEN):
            return _at(day, EXT_OPEN)
        if news_ts + min_left <= _at(day, EXT_CLOSE):
            return news_ts
        i += 1
    if i >= len(trading_days_sorted):
        return None
    return _at(trading_days_sorted[i], EXT_OPEN)


def ext_exit_for_entry(entry: pd.Timestamp) -> pd.Timestamp:
    return min(entry + pd.Timedelta(minutes=config.HORIZON_MINUTES), _at(market_date(entry), EXT_CLOSE))


def is_regular_window(entry: pd.Timestamp, exit_: pd.Timestamp) -> bool:
    """True if the whole window is inside 9:30-16:00 (cheaper to trade than extended hours)."""
    day = market_date(entry)
    return session_open(day) <= entry and exit_ <= session_close(day)
