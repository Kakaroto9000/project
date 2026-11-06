"""
Proves live/market.py rebuilds the training grid (data/processed/grid): for a mix
of test-period articles -- market hours, the last minutes before the close,
pre-market, after hours, weekends and holidays -- it fetches the prices from
Alpaca the way the live app does and compares the model's inputs, the traded
move and the up/flat/down label with what training saved.

The stocks are the live app's watchlist (INDEX_TICKERS + the top
LIVE_TOP_COMPANIES companies) plus some less traded names from the first 300 of the
universe. The grid is stored as float16, so numbers count as matching within
float16 rounding (0.1% relative).

Run (needs ALPACA_API_KEY / ALPACA_SECRET_KEY; a couple of minutes):
    python -m newsmodel.live.check_market
"""

import json
import time

import numpy as np
import pandas as pd

from newsmodel import config, market_time
from newsmodel.live.market import Market
from newsmodel.modeling.grid_dataset import ARTICLE_NAMES, GRID_DIR, PRICE_NAMES
from newsmodel.modeling.universe_dataset import ARTICLE_TABLE_PATH, FUND_WORDS
from newsmodel.collect.universe import UNIVERSE_PATH, latest_daily

PER_KIND = 4
EXTRA_STOCKS = 25  # less traded names from the universe's first 300, outside the watchlist
REL_TOL, ABS_TOL = 1e-3, 1e-4


def pick_articles(art: pd.DataFrame, rng) -> list:
    """A few test-period articles of each kind, by row number."""
    test = art.iloc[int(len(art) * (config.TRAIN_FRACTION + config.VAL_FRACTION)):]
    local = test["news_timestamp"].dt.tz_convert(market_time.MARKET_TZ)
    minutes = local.dt.hour * 60 + local.dt.minute
    s = test["news_session"]
    kinds = {
        "market hours": (s == "market_hours") & (minutes < 15 * 60),
        "first minutes (9:30-9:45)": (s == "market_hours") & (minutes < 9 * 60 + 45),
        "near close (15:30-15:50)": (s == "market_hours") & (minutes >= 15 * 60 + 30) & (minutes < 15 * 60 + 50),
        "too late (15:50-16:00)": (s == "market_hours") & (minutes >= 15 * 60 + 50),
        "pre-market": s == "pre_market",
        "after hours, 16:00-17:00": (s == "after_hours") & (minutes < 17 * 60),
        "after hours, later": (s == "after_hours") & (minutes >= 17 * 60),
        "weekend": s == "weekend",
        "holiday": s == "holiday",
    }
    picked = []
    for kind, mask in kinds.items():
        rows = np.flatnonzero(mask.to_numpy()) + test.index[0]
        picked += [(kind, int(r)) for r in rng.choice(rows, min(PER_KIND, len(rows)), replace=False)]
    return picked


def close(a, b):
    """Equal within float16 rounding, or both missing."""
    both_nan = np.isnan(a) & np.isnan(b)
    return both_nan | (np.abs(a - b) <= REL_TOL * np.abs(b) + ABS_TOL)


def main():
    rng = np.random.default_rng(0)
    art = pd.read_parquet(ARTICLE_TABLE_PATH)
    meta = json.loads((GRID_DIR / "meta.json").read_text())
    stocks = meta["stocks"]
    G = {k: np.load(GRID_DIR / f"{k}.npy", mmap_mode="r") for k in ("prices", "article", "move", "y_main")}
    universe = json.loads(UNIVERSE_PATH.read_text())
    companies = [u["ticker"] for u in universe if not FUND_WORDS.search(u["name"] or "") and u["ticker"] not in config.INDEX_TICKERS]
    watch = [t for t in config.INDEX_TICKERS if t in stocks] + companies[:config.LIVE_TOP_COMPANIES]
    extra = list(rng.choice([t for t in stocks[:300] if t not in watch], EXTRA_STOCKS, replace=False))
    symbols = watch + extra
    cols = np.array([stocks.index(t) for t in symbols])
    print(f"{len(symbols)} stocks: the {len(watch)}-stock watchlist + {len(extra)} less traded ones from the universe's first 300")

    m = Market()
    days = m.trading_days()
    # Alpaca's calendar vs. the trading days training used (SPY's daily bars)
    daily = latest_daily()
    spy_days = set(daily.loc[daily["symbol"] == config.MARKET_TICKER, "date"])
    lo, hi = max(min(spy_days), days[0]), max(spy_days)
    alpaca_days = {d for d in days if lo <= d <= hi}
    spy_days = {d for d in spy_days if lo <= d}
    print(f"Calendar {lo} - {hi}: Alpaca {len(alpaca_days)} trading days, training {len(spy_days)}; "
          f"only Alpaca {sorted(alpaca_days - spy_days)}, only training {sorted(spy_days - alpaca_days)}")
    # Stocks whose training prices stop before the data's end (delisted, e.g. bought out): training
    # kept using their last price, Alpaca has since filled in zero-volume bars -- can't match
    last_day = daily[daily["symbol"].isin(symbols)].groupby("symbol")["date"].max()
    stopped = np.array([last_day.get(t, hi) < sorted(spy_days)[-5] for t in symbols])
    print(f"Stocks whose training prices stop early (left out of 'still trading'): "
          f"{ {t: str(last_day[t]) for t in np.array(symbols)[stopped]} }")
    del daily

    # Today's minute bars are topped up as the delay passes: pieces must add up to one fetch
    day = sorted(alpaca_days)[-5]
    opened = market_time.session_open(day)
    for until in (opened + pd.Timedelta(minutes=47), opened + pd.Timedelta(hours=3, minutes=1), market_time.session_close(day)):
        pieces = m._minute_day(symbols, day, until)
    m._minutes.clear()
    whole = m._minute_day(symbols, day, market_time.session_close(day))
    key = lambda df: df.sort_values(["symbol", "timestamp"]).reset_index(drop=True).astype({"symbol": str})
    print(f"Minute bars of {day} fetched in 3 pieces vs. at once: {len(pieces)} vs {len(whole)} bars, "
          f"identical: {key(pieces).equals(key(whole))}")

    picked = pick_articles(art, rng)
    P_mine, P_grid, A_mine, A_grid, M_mine, M_grid, L_mine, L_grid, kinds = [], [], [], [], [], [], [], [], []
    consistency = []
    times = []
    for kind, r in picked:
        news = pd.Timestamp(art["news_timestamp"].iloc[r])
        entry, exit_ = m.window(news)
        t0 = time.time()
        prices, article = m.inputs(symbols, news, entry, exit_)
        out = m.outcome(symbols, news, entry, exit_)
        times.append(time.time() - t0)
        P_mine.append(prices); P_grid.append(np.asarray(G["prices"][r][cols], np.float64))
        A_mine.append(article.astype(np.float64)); A_grid.append(np.asarray(G["article"][r], np.float64))
        M_mine.append(out["move"]); M_grid.append(np.asarray(G["move"][r][cols], np.float64))
        L_mine.append(np.array([{"down": 0, "flat": 1, "up": 2}.get(x, -1) for x in out["label"]]))
        L_grid.append(np.asarray(G["y_main"][r][cols]))
        kinds.append(kind)
        with np.errstate(invalid="ignore"):
            consistency.append(np.nanmax(np.abs((out["exit_price"] / out["entry_price"] - 1) * 100 - out["raw_move"]), initial=0))
        print(f"  {kind:28s} {news.tz_convert(market_time.MARKET_TZ):%a %Y-%m-%d %H:%M} -> "
              f"{entry.tz_convert(market_time.MARKET_TZ):%a %H:%M}-{exit_.tz_convert(market_time.MARKET_TZ):%H:%M}  "
              f"labels {np.mean(L_mine[-1] == L_grid[-1]):.1%} same  ({times[-1]:.1f}s)", flush=True)

    P_mine, P_grid = np.stack(P_mine), np.stack(P_grid)
    A_mine, A_grid = np.stack(A_mine), np.stack(A_grid)
    M_mine, M_grid, L_mine, L_grid = map(np.stack, (M_mine, M_grid, L_mine, L_grid))

    def report(name, mine, grid):
        ok = close(mine, grid)
        both = np.isfinite(mine) & np.isfinite(grid)
        diff = np.where(both, np.abs(mine - grid), 0)
        live = np.broadcast_to(~stopped, ok.shape) if ok.ndim == 2 else np.ones(ok.shape, bool)
        print(f"  {name:20s} {ok.mean():8.3%} {diff.max():9.5f} | {ok[live].mean():8.3%} {diff[live].max():9.5f} "
              f"{int((np.isnan(mine) & ~np.isnan(grid)).sum()):8d} {int((~np.isnan(mine) & np.isnan(grid)).sum()):8d} {both.mean():6.0%}")
        return ok

    print(f"\n{len(picked)} articles x {len(symbols)} stocks; share matching the grid within float16 rounding, max abs diff:")
    print(f"  {'':20s} {'all stocks':>18s} | {'still trading':>18s} {'NaN here':>8s} {'NaN grid':>8s} {'known':>6s}")
    print("Stock inputs")
    bad = {}
    for j, name in enumerate(PRICE_NAMES):
        bad[name] = ~report(name, P_mine[:, :, j], P_grid[:, :, j])
    print("Article inputs")
    for j, name in enumerate(ARTICLE_NAMES):
        report(name, A_mine[:, j], A_grid[:, j])
    print("Outcome")
    bad["move"] = ~report("move", M_mine, M_grid)
    print(f"  {'label':20s} {np.mean(L_mine == L_grid):8.3%}   "
          f"(known in both: {np.mean((L_mine >= 0) & (L_grid >= 0)):.1%}; known here only {int(((L_mine >= 0) & (L_grid < 0)).sum())}, "
          f"in grid only {int(((L_mine < 0) & (L_grid >= 0)).sum())})")
    print(f"  exit/entry - 1 vs. raw_move: max diff {max(consistency):.2e} %")

    mism = [(kinds[a], pd.Timestamp(art['news_timestamp'].iloc[picked[a][1]]), symbols[s], name, P_mine[a, s, j] if j < 6 else M_mine[a, s],
             P_grid[a, s, j] if j < 6 else M_grid[a, s])
            for j, name in enumerate(list(PRICE_NAMES) + ["move"]) for a, s in zip(*np.nonzero(bad[name]))]
    if mism:
        by_stock = pd.Series([x[2] for x in mism]).value_counts()
        print(f"\nMismatching cells by stock: {by_stock.to_dict()}")
        mism = sorted(mism, key=lambda x: by_stock[x[2]])  # the rarer cases first
        print(f"First mismatches (of {len(mism)}):")
        for k, t, sym, name, a, b in mism[:25]:
            print(f"  {k:28s} {t.tz_convert(market_time.MARKET_TZ):%Y-%m-%d %H:%M} {sym:6s} {name:20s} live {a:10.4f}  grid {b:10.4f}")
    lab = np.nonzero(L_mine != L_grid)
    for a, s in list(zip(*lab))[:10]:
        print(f"  label: {kinds[a]:28s} {symbols[s]:6s} live {L_mine[a, s]} grid {L_grid[a, s]} (move live {M_mine[a, s]:.4f} grid {M_grid[a, s]:.4f})")

    print(f"\nTime per article (inputs + outcome, {len(symbols)} stocks): first {times[0]:.1f}s, "
          f"median {np.median(times):.1f}s, max {max(times):.1f}s")


if __name__ == "__main__":
    main()
