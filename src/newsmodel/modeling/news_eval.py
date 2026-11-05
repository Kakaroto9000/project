"""
Evaluate predictions for every (article, stock) cell of a grid (modeling/grid_dataset.py):
the all-stocks news model (modeling/news_model.py) or any model saved the same way.

    python -m newsmodel.modeling.news_eval --extended --name news_model_ext

Reads <grid>/proba_val_test.npy (or --proba) and writes <grid>/results_<name>.json.

Trading rules, the same for every model:
  - "cutoff": every up/down call at or above a confidence level
  - "top3":   each article's 3 most confident up/down calls (above the level)
  - Bets are DISTINCT: one per (stock, entry time, direction), so many articles that
    lead to the same trade count once.
  - Stocks are hedged with SPY (costs on both legs); the INDEX_TICKERS funds aren't.
    Slippage per side: 0.05% in 9:30-16:00, 0.10% in extended hours (0.02% / 0.05%
    in the "liquid" case).
  - Validation picks the rule and level (at least 300 distinct bets); test is only
    reported. Edges are judged by trading day (bootstrap 95% range), not by trade.

Flaw checks (the last two "too good" results were backtest flaws, so these are always
printed): where the edge comes from -- regular vs. extended hours, news session,
whether the stock traded in the hour before the news, and how heavily traded the stock is.
"""

import argparse
import json
import re

import numpy as np
import pandas as pd

from newsmodel import config
from newsmodel.modeling.dataset import ARTICLES_PATH, DATASET_PATH
from newsmodel.modeling.grid_dataset import EXT_GRID_DIR, GRID_DIR
from newsmodel.modeling.universe_dataset import ARTICLE_TABLE_PATH

SLIP = {"standard": (0.05, 0.10), "liquid": (0.02, 0.05)}  # (regular hours, extended hours) per side
CONF = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
MIN_BETS = 300


def load(grid, proba_path):
    meta = json.loads((grid / "meta.json").read_text())
    rows = json.loads((grid / "proba_rows.json").read_text())
    P = np.load(proba_path).astype(np.float32)
    lo, hi = rows["val"][0], rows["test"][1]
    y = np.asarray(np.load(grid / "y_main.npy", mmap_mode="r")[lo:hi])
    move = np.asarray(np.load(grid / "move.npy", mmap_mode="r")[lo:hi]).astype(np.float32)
    pre = np.asarray(np.load(grid / "prices.npy", mmap_mode="r")[lo:hi, :, 0]).astype(np.float32)
    art = pd.read_parquet(ARTICLE_TABLE_PATH)
    art["regular"] = np.load(grid / "regular.npy") if (grid / "regular.npy").exists() else True
    if grid.name.startswith(EXT_GRID_DIR.name):
        from newsmodel.collect.universe import latest_daily
        from newsmodel.modeling.grid_dataset import ext_windows
        d = latest_daily()
        days = sorted(d.loc[d["symbol"] == config.MARKET_TICKER, "date"].unique())
        e, _, _, _ = ext_windows(art, days)
        art["entry_time"] = pd.to_datetime(np.where(e > 0, e, 0), utc=True)
    else:
        spy = pd.read_parquet(DATASET_PATH, columns=["ticker", "article_id", "entry_timestamp"])
        entry = spy[spy["ticker"] == config.MARKET_TICKER].set_index("article_id")["entry_timestamp"]
        art["entry_time"] = entry.reindex(art["article_id"]).to_numpy()
    art["gidx"] = np.arange(len(art))  # position in universe_articles.parquet
    art = art.iloc[lo:hi].reset_index(drop=True)
    return meta, rows, P, y, move, pre, art, lo


_NAMED = {}


def named_keys(n_stocks):
    """article_idx * n_stocks + company_idx for every pair where the article names the stock
    (modeling/universe_dataset.py's name and ticker matching)."""
    if n_stocks not in _NAMED:
        from newsmodel.modeling.universe_dataset import PAIR_INDEX_PATH
        p = pd.read_parquet(PAIR_INDEX_PATH, columns=["article_idx", "company_idx", "named_count"])
        p = p[p["named_count"] > 0]
        _NAMED[n_stocks] = np.sort(p["article_idx"].to_numpy(np.int64) * n_stocks + p["company_idx"].to_numpy(np.int64))
    return _NAMED[n_stocks]


def liquidity_rank(stocks):
    """Each stock's rank by average daily dollar volume (0 = most traded), from the universe order."""
    return {s: i for i, s in enumerate(stocks)}  # universe.json is sorted by dollar volume


def bets(P, Y, M, pre, art, stocks, funds, min_conf, top, cost):
    pred = P.argmax(-1)
    conf = P.max(-1)
    d = np.where(pred == 2, 1, np.where(pred == 0, -1, 0))
    take = (d != 0) & (conf >= min_conf) & np.isfinite(M) & (Y >= 0)
    if top:
        score = np.where(take, conf, -1)
        keep = np.zeros_like(take)
        np.put_along_axis(keep, np.argsort(-score, axis=1)[:, :top], True, axis=1)
        take &= keep
    a, s = np.nonzero(take)
    cols = ["stock", "entry", "dir", "gross", "cost", "conf", "article", "fund", "session", "regular", "traded_before", "named", "articles", "net", "right", "day"]
    if len(a) == 0:
        return pd.DataFrame(columns=cols)
    names = np.array(stocks)[s]
    is_fund = np.isin(names, funds)
    regular = art["regular"].to_numpy()[a]
    slip = np.where(regular, cost[0], cost[1])
    t = pd.DataFrame({"stock": names, "entry": art["entry_time"].to_numpy()[a], "dir": d[a, s], "gross": d[a, s] * M[a, s],
                      "cost": 2 * slip * np.where(is_fund, 1, 2), "conf": conf[a, s], "article": art["article_id"].to_numpy()[a],
                      "fund": is_fund, "session": art["news_session"].to_numpy()[a], "regular": regular,
                      "traded_before": np.nan_to_num(pre[a, s]) != 0,
                      "named": np.isin(art["gidx"].to_numpy()[a].astype(np.int64) * len(stocks) + s, named_keys(len(stocks)))})
    g = t.sort_values("conf", ascending=False).groupby(["stock", "entry", "dir"], sort=False).agg(
        gross=("gross", "first"), cost=("cost", "first"), conf=("conf", "first"), article=("article", "first"),
        fund=("fund", "first"), session=("session", "first"), regular=("regular", "first"),
        traded_before=("traded_before", "first"), named=("named", "max"), articles=("article", "size")).reset_index()
    g["net"] = g["gross"] - g["cost"]
    g["right"] = g["gross"] > 0
    g["day"] = pd.to_datetime(g["entry"], utc=True).dt.tz_convert("America/New_York").dt.date
    return g


def summ(b):
    if len(b) == 0:
        return {"n": 0, "gross": 0.0, "net": 0.0, "dir_right": 0.0, "days": 0, "articles": 0}
    return {"n": int(len(b)), "gross": float(b["gross"].mean()), "net": float(b["net"].mean()), "dir_right": float(b["right"].mean()),
            "win": float((b["net"] > 0).mean()), "days": int(b["day"].nunique()), "articles": int(b["articles"].sum())}


def daily(b, col="gross"):
    dg = b.groupby("day")[col].mean() if len(b) else pd.Series(dtype=float)
    if len(dg) < 6:
        return (float(dg.mean()) if len(dg) else 0.0), [0.0, 0.0]
    rng = np.random.default_rng(0)
    boot = [rng.choice(dg.values, len(dg)).mean() for _ in range(3000)]
    return float(dg.mean()), [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def report(P, Y, M, pre, art, stocks, funds, way, mc):
    top = 3 if way == "top3" else 0
    known = Y >= 0
    pred = P.argmax(-1)
    out = {"articles": int(len(art)), "from": str(art["news_timestamp"].min()), "to": str(art["news_timestamp"].max()),
           "accuracy": float((pred[known] == Y[known]).mean()), "baseline_flat": float((Y[known] == 1).mean()),
           "labels": {k: float((Y[known] == i).mean()) for i, k in enumerate(["down", "flat", "up"])}, "way": way, "min_conf": mc}
    for i, cls in ((2, "up"), (0, "down")):
        said = (pred == i) & known
        out[f"precision_{cls}"] = float((Y[said] == i).mean()) if said.any() else None
    out["by_conf"] = {}
    for w in ("cutoff", "top3"):
        out["by_conf"][w] = {}
        for c in CONF:
            b = bets(P, Y, M, pre, art, stocks, funds, c, 3 if w == "top3" else 0, SLIP["standard"])
            out["by_conf"][w][str(c)] = {"all": summ(b), "fund": summ(b[b["fund"]]) if len(b) else summ(b),
                                         "stock": summ(b[~b["fund"]]) if len(b) else summ(b)}
    rank = liquidity_rank(stocks)
    for name, cost in SLIP.items():
        b = bets(P, Y, M, pre, art, stocks, funds, mc, top, cost)
        out[f"trades_{name}"] = summ(b)
        eq = b.groupby("day")["net"].mean().cumsum() if len(b) else pd.Series(dtype=float)
        out[f"equity_{name}"] = [[str(k), float(v)] for k, v in eq.items()]
        if name == "standard":
            out["daily_gross_mean"], out["daily_gross_ci95"] = daily(b)
            out["daily_net_mean"], out["daily_net_ci95"] = daily(b, "net")
            if len(b):
                b["liq"] = pd.cut(b["stock"].map(rank), [-1, 99, 299, 1e9], labels=["top 100", "101-300", "301+"])
                out["checks"] = {
                    "window": {k: summ(b[b["regular"] == v]) for k, v in (("9:30-16:00", True), ("extended hours", False))},
                    "session": {s: summ(b[b["session"] == s]) for s in sorted(b["session"].unique())},
                    "traded_before": {k: summ(b[b["traded_before"] == v]) for k, v in (("traded in the hour before", True), ("no trade in the hour before", False))},
                    "liquidity": {str(k): summ(b[b["liq"] == k]) for k in ["top 100", "101-300", "301+"]},
                    "kind": {"fund": summ(b[b["fund"]]), "stock": summ(b[~b["fund"]])},
                    "named": {"named in the article": summ(b[b["named"]]), "not named": summ(b[~b["named"]])},
                }
                per = b.groupby("stock").agg(n=("gross", "size"), gross=("gross", "mean"), right=("right", "mean"))
                out["by_stock"] = per[per["n"] >= 20].sort_values("gross", ascending=False).reset_index().to_dict("records")
            out["_bets"] = b
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extended", action="store_true")
    ap.add_argument("--name", required=True)
    ap.add_argument("--proba", default=None, help="probabilities file (default <grid>/proba_val_test.npy)")
    ap.add_argument("--log", default=None, help="training log, for the loss curve")
    ap.add_argument("--only-named", action="store_true", help="trade only stocks the article names (and funds)")
    ap.add_argument("--only-clean", action="store_true", help="trade only 9:30-16:00 windows and the 300 most traded stocks")
    ap.add_argument("--horizon", type=int, default=None, help="label window in minutes (default HORIZON_MINUTES)")
    ap.add_argument("--top", type=int, default=0, help="score only the funds + this many most traded companies")
    args = ap.parse_args()
    from newsmodel.modeling.grid_dataset import grid_dir
    if args.horizon:
        config.HORIZON_MINUTES = args.horizon
    grid = grid_dir(args.extended, args.horizon)
    meta, rows, P, Y, M, pre, art, lo = load(grid, args.proba or grid / "proba_val_test.npy")
    stocks, funds = meta["stocks"], meta["funds"]
    if args.only_named:  # every other cell becomes "no label", so it's never traded
        keys = art["gidx"].to_numpy(np.int64)[:, None] * len(stocks) + np.arange(len(stocks))[None]
        ok = np.isin(keys, named_keys(len(stocks)))
        ok[:, [stocks.index(f) for f in funds]] = True
        Y = np.where(ok, Y, -1)
    if args.only_clean:
        ok = art["regular"].to_numpy(bool)[:, None] & (np.arange(len(stocks)) < 300)[None]
        Y = np.where(ok, Y, -1)
    if args.top:  # only the live app's watchlist: a --top model, or a full model cut down to compare with one
        from newsmodel.live.predictor import watchlist
        keep = watchlist(stocks, args.top)
        cols = np.array([stocks.index(s) for s in keep])
        Y, M, pre = Y[:, cols], M[:, cols], pre[:, cols]
        if P.shape[1] == len(stocks):
            P = P[:, cols]
        stocks = keep
    v = slice(0, rows["val"][1] - lo)
    t = slice(rows["test"][0] - lo, rows["test"][1] - lo)
    av, at = art.iloc[v].reset_index(drop=True), art.iloc[t].reset_index(drop=True)
    best, score = ("cutoff", 0.5), -1e9
    for way in ("cutoff", "top3"):
        for c in CONF:
            s = summ(bets(P[v], Y[v], M[v], pre[v], av, stocks, funds, c, 3 if way == "top3" else 0, SLIP["standard"]))
            if s["n"] >= MIN_BETS and s["net"] > score:
                best, score = (way, c), s["net"]
    way, mc = best
    val = report(P[v], Y[v], M[v], pre[v], av, stocks, funds, way, mc)
    test = report(P[t], Y[t], M[t], pre[t], at, stocks, funds, way, mc)
    b = test.pop("_bets"); val.pop("_bets")
    heads = pd.read_parquet(ARTICLES_PATH, columns=["article_id", "headline"]).set_index("article_id")["headline"]
    big = pd.concat([b.nlargest(6, "net"), b.nsmallest(6, "net")]) if len(b) else b
    test["examples"] = [{"ticker": r.stock, "kind": "fund" if r.fund else "stock",
                         "when": str(pd.Timestamp(r.entry).tz_convert("America/New_York"))[:16], "pred": "up" if r.dir > 0 else "down",
                         "conf": float(r.conf), "net": float(r.net), "articles": int(r.articles), "regular": bool(r.regular),
                         "headline": str(heads.get(r.article, ""))[:140]} for r in big.itertuples()]
    curve = []
    if args.log:
        log = open(args.log, errors="ignore").read()
        curve = [{"epoch": int(e), "train": float(a), "val": float(x)} for e, a, x in
                 re.findall(r"epoch (\d+) done: training loss ([\d.]+), validation loss ([\d.]+)", log)]
    res = {"name": args.name, "grid": grid.name, "picked": {"way": way, "min_conf": mc}, "val": val, "test": test, "curve": curve}
    (grid / f"results_{args.name}.json").write_text(json.dumps(res, indent=1, default=str))
    for nm, r in (("VAL", val), ("TEST", test)):
        s, l = r["trades_standard"], r["trades_liquid"]
        print(f"{nm}: acc {r['accuracy']:.3f} (always flat {r['baseline_flat']:.3f}) | {way} >= {mc}: {s['n']} distinct bets, "
              f"gross {s['gross']:+.3f}%, net {s['net']:+.3f}% (liquid {l['net']:+.3f}%), right {s['dir_right']:.1%}, "
              f"daily gross {r['daily_gross_mean']:+.3f} {[round(x, 3) for x in r['daily_gross_ci95']]}")
    if "checks" in test:
        for k, groups in test["checks"].items():
            print(f"  TEST by {k}: " + " | ".join(f"{g}: {x['n']} bets {x['gross']:+.3f}%" for g, x in groups.items()))


if __name__ == "__main__":
    main()
