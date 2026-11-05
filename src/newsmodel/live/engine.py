"""
The live loop behind the app: collect news, wait until the free plan's delayed
prices cover each article's trading window, then predict and pretend to trade.

  every LIVE_NEWS_CHECK_SECONDS   new articles from live/news.py -> articles table
                                  ("waiting", with the time its window's prices
                                  will be available: market.ready_at), and all of
                                  them to the training archive (live/archive.py)
  every few seconds               each waiting article whose time has come:
                                    inputs from prices before the news (as training)
                                    -> the model's down/flat/up for every watched stock
                                    -> what actually happened over the window
                                    -> a pretend trade for every "up" call

Pretend trades: LIVE_TRADE_DOLLARS bought at the window's entry price and sold
at its exit price, each side SLIPPAGE_PCT worse, any number at once. No orders
are sent anywhere. Articles that reached us more than LIVE_MAX_LATE_MINUTES into
their window are still predicted, but not traded -- nobody could have.

Runs inside live/server.py (started by the NewsTrader.exe window, ui/) as a
background thread, or on its own:
    python -m newsmodel.live.engine
"""

import json
import threading
import time
import traceback

import numpy as np
import pandas as pd

from newsmodel import config
from newsmodel.keep_awake import keep_awake
from newsmodel.live import archive
from newsmodel.live.market import NotReady
from newsmodel.live.predictor import article_text
from newsmodel.live.store import Store, now_iso

FIRST_LOOKBACK_HOURS = 6      # on the very first start, collect news this far back
NEWS_OVERLAP_MINUTES = 360    # re-ask this far before the newest article: RSS items can show up
                              # an hour or more late (repeats are dropped by uri)
RETRY_MINUTES = 5             # a failed article is retried this much later...
MAX_ATTEMPTS = 3              # ...this many times
ARTICLES_PER_PASS = 5         # then the news is checked again


def iso(ts) -> str:
    return None if ts is None else pd.Timestamp(ts).isoformat()


class Engine(threading.Thread):
    def __init__(self, store: Store):
        super().__init__(daemon=True, name="engine")
        self.store = store
        self.stop_event = threading.Event()
        self.state = "starting"          # shown in the app's status bar
        self.check_now = threading.Event()
        self.predictor = self.market = None

    def stop(self):
        self.stop_event.set()

    def run(self):
        with keep_awake():
            try:
                self.state = "loading the model"
                from newsmodel.live.market import Market
                from newsmodel.live.predictor import Predictor
                self.market = Market()
                self.predictor = Predictor()
                self.store.log(f"Model {self.predictor.name} loaded on {self.predictor.device}: "
                               f"{len(self.predictor.stocks)} stocks watched")
            except Exception as e:
                self.state = f"failed to start: {e}"
                self.store.log(f"Engine failed to start: {e}\n{traceback.format_exc()}")
                return
            next_news = 0.0
            while not self.stop_event.is_set():
                try:
                    if time.time() >= next_news or self.check_now.is_set():
                        self.check_now.clear()
                        self.state = "checking news"
                        self.collect_news()
                        next_news = time.time() + config.LIVE_NEWS_CHECK_SECONDS
                    self.state = "predicting"
                    self.process_ready()
                    self.state = "running"
                except Exception as e:
                    self.state = f"error: {e}"
                    self.store.log(f"Error: {e}\n{traceback.format_exc()}")
                self.stop_event.wait(5)
        self.state = "stopped"

    # --- News ---
    def collect_news(self):
        from newsmodel.live import news
        now = pd.Timestamp.now(tz="UTC")
        since = self.store.get("news_since")
        since = pd.Timestamp(since) if since else now - pd.Timedelta(hours=FIRST_LOOKBACK_HOURS)
        items = news.fetch_new(since - pd.Timedelta(minutes=NEWS_OVERLAP_MINUTES))
        known = {r["uri"] for r in self.store.rows("SELECT uri FROM articles WHERE published >= ?",
                                                   (iso(since - pd.Timedelta(days=1)),))}
        watched = set(self.predictor.stocks)
        new, fresh = 0, []
        for a in items:
            if a["uri"] in known:
                continue
            known.add(a["uri"])
            fresh.append(a)
            status, note, entry, exit_, ready = "waiting", None, None, None, None
            symbols = a.get("symbols") or []
            if config.LIVE_WATCHLIST_NEWS_ONLY and symbols and not watched & set(symbols):
                status, note = "skipped", "only about stocks outside the watchlist"
            else:
                entry, exit_ = self.market.window(a["published"])
                if entry is not None:
                    ready = self.market.ready_at(a["published"])
            self.store.run("INSERT OR IGNORE INTO articles (uri, title, body, source, url, symbols, published, received, "
                           "entry, exit, ready, status, note) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                           (a["uri"], a["title"], a["body"], a["source"], a["url"], json.dumps(symbols),
                            iso(a["published"]), now_iso(), iso(entry), iso(exit_), iso(ready), status, note))
            new += status == "waiting"
        archive.save(fresh, now)  # every article, skipped or not, kept for training later
        if items:
            newest = max(pd.Timestamp(a["published"]) for a in items)
            self.store.put("news_since", iso(max(newest, since)))
        self.store.put("news_checked", now_iso())
        if new:
            self.store.log(f"{new} new article(s) waiting for their window's prices")

    # --- Predict and pretend-trade ---
    def process_ready(self):
        now = now_iso()
        due = self.store.rows("SELECT * FROM articles WHERE status = 'waiting' AND (ready IS NULL OR ready <= ?) "
                              "ORDER BY published LIMIT ?", (now, ARTICLES_PER_PASS))
        for a in due:
            if self.stop_event.is_set() or self.check_now.is_set():
                return
            if a["ready"] is None:  # the window wasn't known yet (e.g. past the calendar): try again
                ready = self.market.ready_at(pd.Timestamp(a["published"]))
                entry, exit_ = self.market.window(pd.Timestamp(a["published"]))
                if ready is None or entry is None:
                    continue
                self.store.run("UPDATE articles SET entry = ?, exit = ?, ready = ? WHERE uri = ?",
                               (iso(entry), iso(exit_), iso(ready), a["uri"]))
                if iso(ready) > now:
                    continue
                a["entry"], a["exit"] = iso(entry), iso(exit_)
            try:
                self.process(a)
            except NotReady:  # the delayed prices don't reach the window's end yet: just wait
                later = pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=1)
                self.store.run("UPDATE articles SET ready = ? WHERE uri = ?", (iso(later), a["uri"]))
            except Exception as e:
                attempts = (a["attempts"] or 0) + 1
                give_up = attempts >= MAX_ATTEMPTS
                retry = pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=RETRY_MINUTES)
                self.store.run("UPDATE articles SET attempts = ?, status = ?, ready = ?, note = ? WHERE uri = ?",
                               (attempts, "error" if give_up else "waiting", iso(retry), str(e)[:300], a["uri"]))
                self.store.log(f"Article failed ({attempts}/{MAX_ATTEMPTS}): {a['title'][:80]}: {e}")

    def process(self, a: dict):
        stocks = self.predictor.stocks
        news_ts, entry, exit_ = pd.Timestamp(a["published"]), pd.Timestamp(a["entry"]), pd.Timestamp(a["exit"])
        prices, article = self.market.inputs(stocks, news_ts, entry, exit_)
        probs = self.predictor.predict(article_text(a["title"], a["body"]), prices, article)
        out = self.market.outcome(stocks, news_ts, entry, exit_)
        buy = (probs.argmax(1) == 2) & (probs[:, 2] >= config.LIVE_MIN_UP_PROB)
        late = (pd.Timestamp(a["received"]) - entry).total_seconds() / 60
        tradable = late <= config.LIVE_MAX_LATE_MINUTES

        num = lambda v: None if v is None or not np.isfinite(v) else float(v)
        preds, trades = [], []
        slip = config.SLIPPAGE_PCT / 100
        for i, t in enumerate(stocks):
            ep, xp = out["entry_price"][i], out["exit_price"][i]
            preds.append((a["uri"], t, *map(float, probs[i]), int(buy[i]), num(ep), num(xp), num(out["raw_move"][i]),
                          num(out["move"][i]), num(out["threshold"][i]), out["label"][i]))
            if tradable and buy[i] and np.isfinite(ep) and np.isfinite(xp):
                bp, sp = ep * (1 + slip), xp * (1 - slip)
                qty = config.LIVE_TRADE_DOLLARS / bp
                trades.append((a["uri"], t, float(probs[i, 2]), a["entry"], bp, qty, a["exit"], sp,
                               qty * (sp - bp), (sp / bp - 1) * 100))
        self.store.many("INSERT OR REPLACE INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", preds)
        self.store.many("INSERT OR IGNORE INTO trades (uri, ticker, p_up, buy_time, buy_price, qty, sell_time, sell_price, "
                        "pnl, pnl_pct) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", trades)
        no_price = int(np.sum(buy & ~(np.isfinite(out["entry_price"]) & np.isfinite(out["exit_price"]))))
        note = (f"reached us {late:.0f} min into its window: not traded" if not tradable
                else f"{no_price} up call(s) without a price to trade" if no_price else None)
        self.store.run("UPDATE articles SET status = 'done', processed = ?, buys = ?, note = ? WHERE uri = ?",
                       (now_iso(), len(trades), note, a["uri"]))
        pnl = sum(t[8] for t in trades)
        self.store.log(f"Predicted: {a['title'][:80]} -> {len(trades)} pretend buy(s), P&L ${pnl:+,.2f}")


def main():
    store = Store()
    engine = Engine(store)
    engine.start()
    print("Engine running; Ctrl+C to stop. Log:", config.LIVE_DIR / "engine.log")
    try:
        while engine.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        engine.stop()
        engine.join(10)


if __name__ == "__main__":
    main()
