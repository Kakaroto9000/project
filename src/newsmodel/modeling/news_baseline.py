"""
The comparison for the all-stocks news model: the same grid and labels, but NO article
text -- LightGBM on each (article, stock) cell's price and timing inputs plus which stock
it is. If the language model can't beat this, the text isn't adding anything.

    python -m newsmodel.modeling.news_baseline --extended

Trains on a weighted sample of training cells (each label window counts once, like the
news model), stops early on a validation sample, then predicts every known validation and
test cell. Saves <grid>/proba_val_test_no_text.npy in the same layout as the news model,
so modeling/news_eval.py scores both the same way.
"""

import argparse
import json
import time

import numpy as np

from newsmodel import config
from newsmodel.keep_awake import keep_awake
from newsmodel.modeling.grid_dataset import EXT_GRID_DIR, GRID_DIR

SAMPLE_CELLS = 4_000_000


def cells(grid, rows, n_s):
    """Feature matrix for (article rows x all stocks): 6 price inputs, 9 article inputs, stock id."""
    prices = np.load(grid / "prices.npy", mmap_mode="r")
    article = np.load(grid / "article.npy")
    p = np.asarray(prices[rows]).astype(np.float32).reshape(-1, prices.shape[-1])
    a = np.repeat(article[rows], n_s, axis=0)
    s = np.tile(np.arange(n_s, dtype=np.float32), len(rows))[:, None]
    return np.concatenate([p, a, s], axis=1)


def main():
    import lightgbm as lgb
    ap = argparse.ArgumentParser()
    ap.add_argument("--extended", action="store_true")
    ap.add_argument("--horizon", type=int, default=None, help="label window in minutes (default HORIZON_MINUTES)")
    args = ap.parse_args()
    from newsmodel.modeling.grid_dataset import grid_dir
    if args.horizon:
        config.HORIZON_MINUTES = args.horizon
    grid = grid_dir(args.extended, args.horizon)
    t0 = time.time()
    meta = json.loads((grid / "meta.json").read_text())
    y = np.load(grid / "y_main.npy", mmap_mode="r")
    weight = np.load(grid / "weight.npy")
    n_a, n_s = y.shape
    cut1, cut2 = int(n_a * config.TRAIN_FRACTION), int(n_a * (config.TRAIN_FRACTION + config.VAL_FRACTION))
    rng = np.random.default_rng(0)
    names = meta["price_names"] + meta["article_names"] + ["stock"]

    def sample(lo, hi, n):
        arts = np.sort(rng.choice(np.arange(lo, hi), min(hi - lo, n // n_s * 3), replace=False))
        X = cells(grid, arts, n_s)
        Y = np.asarray(y[arts]).reshape(-1)
        W = np.repeat(weight[arts], n_s)
        ok = Y >= 0
        X, Y, W = X[ok], Y[ok], W[ok]
        pick = rng.choice(len(Y), min(n, len(Y)), replace=False)
        counts = np.bincount(Y[pick], minlength=3)
        W = W[pick] * (len(pick) / (3 * np.maximum(counts, 1)))[Y[pick]]  # balanced classes
        return X[pick], Y[pick], W

    with keep_awake():
        Xtr, Ytr, Wtr = sample(0, cut1, SAMPLE_CELLS)
        Xva, Yva, Wva = sample(cut1, cut2, SAMPLE_CELLS // 5)
        print(f"{len(Ytr):,} training cells, {len(Yva):,} validation cells ({time.time() - t0:.0f}s)", flush=True)
        params = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=63, min_data_in_leaf=2000,
                      feature_fraction=0.9, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, num_threads=16, verbose=-1, seed=0)
        dtr = lgb.Dataset(Xtr, Ytr, weight=Wtr, feature_name=names, categorical_feature=["stock"])
        dva = lgb.Dataset(Xva, Yva, weight=Wva, reference=dtr)
        booster = lgb.train(params, dtr, 600, valid_sets=[dva], callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(50)])
        print(f"trained, {booster.best_iteration} trees ({time.time() - t0:.0f}s)", flush=True)
        gain = booster.feature_importance("gain")
        imp = sorted(zip(names, (gain / gain.sum()).tolist()), key=lambda x: -x[1])
        out = np.full((n_a - cut1, n_s, 3), 1 / 3, np.float16)
        for i in range(cut1, n_a, 2000):
            arts = np.arange(i, min(i + 2000, n_a))
            X = cells(grid, arts, n_s)
            known = np.asarray(y[arts]).reshape(-1) >= 0
            p = np.full((len(X), 3), 1 / 3, np.float32)
            p[known] = booster.predict(X[known], num_iteration=booster.best_iteration)
            out[i - cut1:i - cut1 + len(arts)] = p.reshape(len(arts), n_s, 3).astype(np.float16)
            if (i - cut1) % 10000 == 0:
                print(f"  predicted {i - cut1 + len(arts):,}/{n_a - cut1:,} articles ({time.time() - t0:.0f}s)", flush=True)
    np.save(grid / "proba_val_test_no_text.npy", out)
    (grid / "no_text_importance.json").write_text(json.dumps(imp))
    print(f"Saved {out.shape} to {grid / 'proba_val_test_no_text.npy'} ({time.time() - t0:.0f}s)")
    print("top inputs:", [(n, round(g, 3)) for n, g in imp[:8]])


if __name__ == "__main__":
    main()
