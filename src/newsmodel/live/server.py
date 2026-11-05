"""
The live engine (live/engine.py) plus a small local web API for the Tauri
window (ui/). Started by the window itself; only listens on 127.0.0.1.

  GET  /api/status         engine state, counts, settings, and a `version` that
                           changes whenever there's something new to show
  GET  /api/now            incoming news, the latest two predicted articles, today's numbers
  GET  /api/article?uri=   one article: text, every stock's prediction and outcome, its trades
  GET  /api/history        every article with its pretend buys, and the overall scorecard
  GET  /api/portfolio      the pretend account over time and every trade
  GET  /api/log            the engine's recent log
  POST /api/check          check the news now instead of at the next minute

Run:  python -m newsmodel.live.server   [--port 8765] [--no-engine] [--db other.db]
"""

import argparse
import math
import sys

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from newsmodel import config
from newsmodel.live.engine import Engine
from newsmodel.live.store import Store

PORT = 8765
TZ = "America/New_York"
CALLS = np.array(["down", "flat", "up"])

store = Store()
engine = Engine(store)
app = FastAPI(title="News Trader")
# The Tauri window (tauri://localhost on Windows is served as http://tauri.localhost) and the Vite dev server
app.add_middleware(CORSMiddleware, allow_origins=["http://tauri.localhost", "https://tauri.localhost", "tauri://localhost",
                                                  "http://localhost:1420"], allow_methods=["*"], allow_headers=["*"])


def clean(v):
    """JSON-safe: NaN -> None, numpy numbers -> Python numbers, recursively."""
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, (np.floating, float)):
        return None if math.isnan(v) or math.isinf(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def records(df: pd.DataFrame) -> list:
    return clean(df.to_dict("records"))


def today_mask(times: pd.Series) -> np.ndarray:
    return (pd.to_datetime(times, utc=True).dt.tz_convert(TZ).dt.date == pd.Timestamp.now(tz=TZ).date()).to_numpy()


def trades_with_outcome() -> pd.DataFrame:
    return store.frame("SELECT t.*, p.actual, p.move, a.title, a.source FROM trades t "
                       "JOIN predictions p ON p.uri = t.uri AND p.ticker = t.ticker JOIN articles a ON a.uri = t.uri "
                       "ORDER BY t.sell_time")


@app.get("/api/status")
def status():
    c = {r["status"]: r["n"] for r in store.rows("SELECT status, COUNT(*) n FROM articles GROUP BY status")}
    last_log = store.rows("SELECT MAX(id) m FROM log")[0]["m"] or 0
    nxt = store.rows("SELECT MIN(ready) r FROM articles WHERE status = 'waiting' AND ready IS NOT NULL")[0]["r"]
    return clean({
        "engine": engine.state, "news_checked": store.get("news_checked"), "counts": c, "next_ready": nxt,
        "version": f"{sum(c.values())}-{c.get('done', 0)}-{c.get('waiting', 0)}-{last_log}",
        "model": config.LIVE_MODEL_PATH.name, "sources": list(getattr(config, "LIVE_NEWS_SOURCES", [])),
        "settings": {"start_cash": config.LIVE_START_CASH, "trade_dollars": config.LIVE_TRADE_DOLLARS,
                     "hold_minutes": config.LIVE_HOLD_MINUTES, "slippage_pct": config.SLIPPAGE_PCT,
                     "min_up_prob": config.LIVE_MIN_UP_PROB, "max_late_minutes": config.LIVE_MAX_LATE_MINUTES},
    })


@app.post("/api/check")
def check():
    engine.check_now.set()
    return {"ok": True}


@app.get("/api/now")
def now():
    waiting = store.frame("SELECT uri, published, source, title, ready, symbols FROM articles WHERE status = 'waiting' "
                          "ORDER BY published DESC LIMIT 200")
    done = store.rows("SELECT uri FROM articles WHERE status = 'done' ORDER BY published DESC, processed DESC LIMIT 2")
    t = trades_with_outcome()
    td = t[today_mask(t["sell_time"])] if len(t) else t
    total = float(t["pnl"].sum()) if len(t) else 0.0
    return clean({
        "incoming": records(waiting),
        "latest": done[0]["uri"] if done else None,
        "previous": done[1]["uri"] if len(done) > 1 else None,
        "account": config.LIVE_START_CASH + total, "total_pnl": total,
        "today_pnl": float(td["pnl"].sum()) if len(td) else 0.0, "today_trades": len(td),
        "today_hit": float((td["actual"] == "up").mean()) if len(td) else None,
        # the last day of account values, for the small chart on the Now page
        "spark": spark(t),
    })


def spark(t: pd.DataFrame) -> list:
    if t.empty:
        return []
    ms = (pd.to_datetime(t["sell_time"], utc=True).astype("int64") // 10**6).to_numpy()
    value = config.LIVE_START_CASH + t["pnl"].cumsum().to_numpy()
    return [[int(a), float(b)] for a, b in zip(ms[-400:], value[-400:])]


@app.get("/api/article")
def article(uri: str):
    rows = store.rows("SELECT * FROM articles WHERE uri = ?", (uri,))
    if not rows:
        raise HTTPException(404, "no such article")
    a = rows[0]
    a["body"] = (a["body"] or "")[:4000]
    p = store.frame("SELECT * FROM predictions WHERE uri = ?", (uri,))
    t = store.frame("SELECT ticker, buy_price, sell_price, qty, pnl, pnl_pct FROM trades WHERE uri = ?", (uri,))
    summary = None
    if len(p):
        p["call"] = CALLS[p[["p_down", "p_flat", "p_up"]].to_numpy().argmax(1)]
        p = p.sort_values("p_up", ascending=False)
        moved = p["actual"].notna() & (p["call"] != "flat")
        buys = p[p["buy"] == 1]
        summary = {"buys": len(t), "pnl": float(t["pnl"].sum()) if len(t) else 0.0,
                   "buys_right": int((buys["actual"] == "up").sum()), "buys_known": int(buys["actual"].notna().sum()),
                   "calls_right": int((p.loc[moved, "call"] == p.loc[moved, "actual"]).sum()), "calls_known": int(moved.sum())}
    return clean({"article": a, "predictions": records(p), "trades": records(t), "summary": summary})


@app.get("/api/history")
def history():
    arts = store.frame("SELECT uri, published, source, title, status, note FROM articles "
                       "WHERE status != 'skipped' ORDER BY published DESC LIMIT 3000")
    t = trades_with_outcome()
    if len(t):
        by = t.groupby("uri").agg(buys=("pnl", "size"), right=("actual", lambda s: int((s == "up").sum())),
                                  avg=("pnl_pct", "mean"), pnl=("pnl", "sum"))
        arts = arts.join(by, on="uri")
    p = store.frame("SELECT p_down, p_flat, p_up, buy, actual, move FROM predictions WHERE actual IS NOT NULL")
    card = {"articles": int((arts["status"] == "done").sum()), "pending": int((arts["status"] != "done").sum())}
    if len(p):
        call = CALLS[p[["p_down", "p_flat", "p_up"]].to_numpy().argmax(1)]
        moved = call != "flat"
        b = p[p["buy"] == 1]
        card |= {"calls": int(moved.sum()), "calls_right": float((call[moved] == p["actual"].to_numpy()[moved]).mean()) if moved.any() else None,
                 "buys": len(b), "buys_right": float((b["actual"] == "up").mean()) if len(b) else None,
                 "avg_buy_move": float(b["move"].mean()) if len(b) else None, "avg_all_move": float(p["move"].mean())}
    card["pnl"] = float(t["pnl"].sum()) if len(t) else 0.0
    return clean({"articles": records(arts), "card": card})


@app.get("/api/portfolio")
def portfolio():
    t = trades_with_outcome()
    if t.empty:
        return clean({"stats": None, "equity": [], "trades": []})
    value = config.LIVE_START_CASH + t["pnl"].cumsum()
    today = today_mask(t["sell_time"])
    first = pd.Timestamp(t["buy_time"].min()).value // 10**6
    equity = [[int(first), float(config.LIVE_START_CASH)]] + spark_all(t, value)
    stats = {"value": float(value.iloc[-1]), "total_pnl": float(t["pnl"].sum()), "today_pnl": float(t.loc[today, "pnl"].sum()),
             "today_trades": int(today.sum()), "trades": len(t), "winners": float((t["pnl"] > 0).mean()),
             "avg_pct": float(t["pnl_pct"].mean()), "avg_pnl": float(t["pnl"].mean()),
             "best": records(t.nlargest(1, "pnl")[["ticker", "pnl", "pnl_pct"]])[0],
             "worst": records(t.nsmallest(1, "pnl")[["ticker", "pnl", "pnl_pct"]])[0]}
    by_ticker = (t.groupby("ticker").agg(trades=("pnl", "size"), pnl=("pnl", "sum")).sort_values("pnl")
                 .reset_index())
    cols = ["id", "uri", "ticker", "title", "source", "p_up", "buy_time", "buy_price", "sell_time", "sell_price", "qty",
            "pnl", "pnl_pct", "actual"]
    return clean({"stats": stats, "equity": equity, "trades": records(t[cols].iloc[::-1].head(3000)),
                  "by_ticker": records(pd.concat([by_ticker.head(8), by_ticker.tail(8)]).drop_duplicates("ticker"))})


def spark_all(t, value) -> list:
    ms = (pd.to_datetime(t["sell_time"], utc=True).astype("int64") // 10**6).to_numpy()
    return [[int(a), float(b)] for a, b in zip(ms, value.to_numpy())]


@app.get("/api/log")
def log():
    return store.rows("SELECT time, message FROM log ORDER BY id DESC LIMIT 400")


def main():
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--no-engine", action="store_true", help="serve the saved data only (for working on the UI)")
    ap.add_argument("--db", help="another SQLite file (e.g. test data)")
    args = ap.parse_args()
    global store, engine
    if args.db:
        from pathlib import Path
        store = Store(Path(args.db))
        engine = Engine(store)
    if sys.stdout is None:  # started without a console by the window: keep the output in a file
        config.LIVE_DIR.mkdir(parents=True, exist_ok=True)
        sys.stdout = sys.stderr = open(config.LIVE_DIR / "server_output.log", "a", encoding="utf-8", buffering=1)
    if not args.no_engine:
        engine.start()
    else:
        engine.state = "off (--no-engine)"
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
