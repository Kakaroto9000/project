"""
Everything the live app knows, in one SQLite file (config.LIVE_DB_PATH), so the
window and the trading engine share it and nothing is lost when the app closes.

  articles     one row per article: text, when it came out, when it reached us,
               its trading window, and where it is: waiting (for the delayed
               prices to cover its window), done, skipped, error
  predictions  one row per (article, stock): the model's down/flat/up and what
               actually happened over the window
  trades       one row per pretend trade: bought at the window's entry price,
               sold at its exit price, LIVE_TRADE_DOLLARS each, SLIPPAGE_PCT each way
  settings     small values (switches, last news check)
  log          what the engine did, for the Log tab

Times are stored as ISO strings in UTC.
"""

import json
import sqlite3
import threading

import pandas as pd

from newsmodel import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    uri TEXT PRIMARY KEY, title TEXT, body TEXT, source TEXT, url TEXT, symbols TEXT,
    published TEXT, received TEXT,
    entry TEXT, exit TEXT, ready TEXT,    -- the trading window, and when delayed prices cover it
    status TEXT, note TEXT, processed TEXT, buys INTEGER, attempts INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS predictions (
    uri TEXT, ticker TEXT, p_down REAL, p_flat REAL, p_up REAL, buy INTEGER,
    entry_price REAL, exit_price REAL, raw_move REAL, move REAL, threshold REAL, actual TEXT,
    PRIMARY KEY (uri, ticker)
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT, uri TEXT, ticker TEXT, p_up REAL,
    buy_time TEXT, buy_price REAL, qty REAL, sell_time TEXT, sell_price REAL,
    pnl REAL, pnl_pct REAL, UNIQUE (uri, ticker)
);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS log (id INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT, message TEXT);
CREATE INDEX IF NOT EXISTS articles_published ON articles (published);
CREATE INDEX IF NOT EXISTS articles_ready ON articles (status, ready);
CREATE INDEX IF NOT EXISTS trades_sell ON trades (sell_time);
"""


def now_iso() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


class Store:
    def __init__(self, path=None):
        path = path or config.LIVE_DB_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")  # the window reads while the engine writes
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()

    def run(self, sql, params=()):
        with self.lock, self.db:
            return self.db.execute(sql, params)

    def many(self, sql, rows):
        with self.lock, self.db:
            self.db.executemany(sql, rows)

    def rows(self, sql, params=()) -> list:
        with self.lock:
            return [dict(r) for r in self.db.execute(sql, params).fetchall()]

    def frame(self, sql, params=()) -> pd.DataFrame:
        with self.lock:
            return pd.read_sql_query(sql, self.db, params=params)

    # --- small helpers ---
    def get(self, key, default=None):
        r = self.rows("SELECT value FROM settings WHERE key = ?", (key,))
        return json.loads(r[0]["value"]) if r else default

    def put(self, key, value):
        self.run("INSERT OR REPLACE INTO settings VALUES (?, ?)", (key, json.dumps(value)))

    def log(self, message: str):
        t = now_iso()
        self.run("INSERT INTO log (time, message) VALUES (?, ?)", (t, message))
        line = f"{pd.Timestamp(t).tz_convert('America/New_York'):%Y-%m-%d %H:%M:%S} {message}"
        with open(config.LIVE_DIR / "engine.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
