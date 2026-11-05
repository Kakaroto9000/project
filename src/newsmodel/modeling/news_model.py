"""
One model that reads an article and predicts all UNIVERSE_SIZE stocks at once --
no tickers, no tags. It learns from modeling/grid_dataset.py's labels which
stocks each kind of news actually moves.

    article --> fine-tuned reader (FINETUNE_MODEL) --> what the article means
                                                          |
    for each stock: its learned profile + its prices --> shared layer --> its own head --> up / flat / down

  - The reader is read once per article; every layer is trained on our labels.
  - Each stock has a learned profile (NEWS_MODEL_PROFILE numbers), which ends
    up close to stocks that react to the same news.
  - Each stock has its own head on top of a shared one: the shared head learns
    what holds for all stocks, the stock's own head (starting at zero) only
    how that stock differs, so rarely covered stocks lean on the shared one.
  - Two answers per stock: the traded one (entry -> exit) and a practice one
    (10 minutes before the article -> exit, training only) that helps it learn
    which news matters. Only the traded answer is ever used for decisions.

Split by article time like the other models. Keeps the epoch with the lowest
validation loss. Saves models/news_model.pt and the traded answer's
probabilities for validation and test (data/processed/grid/proba_val_test.npy).

Run (GPU; ~30 min per epoch on an RTX 4070, nothing else using the GPU):
    python -m newsmodel.modeling.news_model
"""

import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from newsmodel import config
from newsmodel.keep_awake import keep_awake
from newsmodel.modeling.dataset import ARTICLES_PATH
from newsmodel.modeling.grid_dataset import EXT_GRID_DIR, GRID_DIR
from newsmodel.modeling.universe_dataset import ARTICLE_TABLE_PATH

MODEL_PATH = config.MODELS_DIR / "news_model.pt"
AUX_WEIGHT = 0.5  # how much the practice answer counts in training


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


class Inputs:
    """Price and article numbers -> model inputs: signed log, then standardized with
    training-set means; a missing number becomes 0 plus a 'missing' flag."""

    def fit(self, prices, article, train_rows, rng):
        sample = prices[rng.choice(train_rows, min(20000, len(train_rows)), replace=False)].astype(np.float32)
        s = signed_log(sample.reshape(-1, sample.shape[-1]))
        self.p_mean, self.p_std = np.nanmean(s, 0), np.nanstd(s, 0) + 1e-6
        a = signed_log(article[train_rows])
        self.a_mean, self.a_std = np.nanmean(a, 0), np.nanstd(a, 0) + 1e-6
        return self

    def prices(self, p):
        s = (signed_log(p.astype(np.float32)) - self.p_mean) / self.p_std
        miss = np.isnan(s)
        return np.concatenate([np.where(miss, 0, s), miss], axis=-1).astype(np.float32)

    def article(self, a):
        return np.nan_to_num((signed_log(a) - self.a_mean) / self.a_std).astype(np.float32)


class NewsModel(nn.Module):
    def __init__(self, base, n_stocks, n_price, n_article, hidden=256, profile=None):
        super().__init__()
        from transformers import AutoModel
        profile = profile or config.NEWS_MODEL_PROFILE
        self.encoder = AutoModel.from_pretrained(base)
        h = self.encoder.config.hidden_size
        self.article = nn.Linear(h + n_article, hidden)
        self.profile = nn.Embedding(n_stocks, profile)
        self.profile_in = nn.Linear(profile, hidden, bias=False)
        self.price_in = nn.Linear(n_price, hidden, bias=False)
        self.drop = nn.Dropout(0.1)
        self.shared_out = nn.Linear(hidden, 6)                        # 3 traded + 3 practice
        self.own_w = nn.Parameter(torch.zeros(n_stocks, hidden, 6))   # each stock's own head
        self.own_b = nn.Parameter(torch.zeros(n_stocks, 6))

    def forward(self, ids, mask, article_x, price_x):
        t = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        meaning = (t * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        a = self.article(torch.cat([meaning, article_x], 1))               # (B, H)
        s = self.profile_in(self.profile.weight)                           # (S, H)
        z = nn.functional.gelu(a[:, None, :] + s[None] + self.price_in(price_x))  # (B, S, H)
        z = self.drop(z)
        out = self.shared_out(z) + torch.einsum("bsh,sho->bso", z, self.own_w) + self.own_b
        return out[..., :3], out[..., 3:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=config.NEWS_MODEL_EPOCHS)
    ap.add_argument("--limit", type=int, default=0, help="train on this many articles only (quick test)")
    ap.add_argument("--extended", action="store_true", help="use the extended-hours grid (pre-market + after hours)")
    ap.add_argument("--clean-only", action="store_true",
                    help="learn only from windows fully inside 9:30-16:00 and the 300 most traded stocks")
    ap.add_argument("--named-only", action="store_true",
                    help="learn only from stocks the article names (plus the funds): a company-news model")
    ap.add_argument("--horizon", type=int, default=None, help="label window in minutes (default HORIZON_MINUTES)")
    ap.add_argument("--top", type=int, default=0,
                    help="learn only the funds + this many most traded companies (the live app's watchlist)")
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="train on only the newest part of the training articles (as if there were less history)")
    args = ap.parse_args()
    from newsmodel.modeling.grid_dataset import grid_dir
    if args.horizon:
        config.HORIZON_MINUTES = args.horizon
    grid = grid_dir(args.extended, args.horizon)
    tag = ("_ext" if args.extended else "") + ("_named" if args.named_only else "") + ("_clean" if args.clean_only else "") + (f"_{args.horizon}" if args.horizon and args.horizon != 30 else "")
    frac_tag = f"_f{round(args.fraction * 100)}" if args.fraction < 1 else ""
    top_tag = f"_top{args.top}" if args.top else ""
    tag += frac_tag + top_tag
    model_path = config.MODELS_DIR / f"news_model{tag}.pt"
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    t0 = time.time()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    meta = json.loads((grid / "meta.json").read_text())
    y_main = np.load(grid / "y_main.npy", mmap_mode="r")
    y_aux = np.load(grid / "y_aux.npy", mmap_mode="r")
    prices = np.load(grid / "prices.npy", mmap_mode="r")
    article = np.load(grid / "article.npy")
    # Each label window counts once: articles that share one split its weight (1 if not saved)
    art_weight = np.load(grid / "weight.npy") if (grid / "weight.npy").exists() else np.ones(len(article), np.float32)
    if args.top:  # only the watchlist's columns, small enough to hold in memory
        from newsmodel.live.predictor import watchlist
        if args.named_only or args.clean_only:
            raise SystemExit("--top can't be combined with --named-only or --clean-only")
        stocks = watchlist(meta["stocks"], args.top)
        cols = np.array([meta["stocks"].index(t) for t in stocks])
        take = lambda a: np.concatenate([np.asarray(a[i:i + 20000][:, cols]) for i in range(0, len(a), 20000)])
        y_main, y_aux, prices = take(y_main), take(y_aux), take(prices)
        meta["stocks"] = stocks
        print(f"--top {args.top}: {len(stocks)} stocks ({time.time() - t0:.0f}s)", flush=True)
    n_a, n_s = y_main.shape
    named = None
    if args.named_only:  # (article, stock) cells the model may learn from: named stocks + funds
        from newsmodel.modeling.universe_dataset import PAIR_INDEX_PATH
        pi = pd.read_parquet(PAIR_INDEX_PATH, columns=["article_idx", "company_idx", "named_count"])
        pi = pi[pi["named_count"] > 0]
        named = np.zeros((n_a, n_s), bool)
        named[pi["article_idx"].to_numpy(), pi["company_idx"].to_numpy()] = True
        named[:, [meta["stocks"].index(f) for f in meta["funds"]]] = True
        print(f"named-only: {named.mean():.2%} of cells ({named.sum():,})", flush=True)
    if args.clean_only:  # regular-hours windows x the 300 most traded stocks (universe.json is sorted by dollar volume)
        regular = np.load(grid / "regular.npy")
        named = np.zeros((n_a, n_s), bool)
        named[np.ix_(regular, np.arange(n_s) < 300)] = True
        print(f"clean-only: {regular.mean():.0%} of articles in regular hours, {named.mean():.2%} of cells", flush=True)
    cut1, cut2 = int(n_a * config.TRAIN_FRACTION), int(n_a * (config.TRAIN_FRACTION + config.VAL_FRACTION))
    tr, va, te = np.arange(cut1), np.arange(cut1, cut2), np.arange(cut2, n_a)
    if args.limit:
        tr, va, te = tr[:args.limit], va[:args.limit // 5], te[:args.limit // 5]
    if args.fraction < 1:  # drop the oldest articles; validation and test stay the same
        tr = tr[len(tr) - int(len(tr) * args.fraction):]
    print(f"{len(tr)} train / {len(va)} validation / {len(te)} test articles x {n_s} stocks ({time.time() - t0:.0f}s)", flush=True)

    inputs = Inputs().fit(prices, article, tr, rng)
    art = pd.read_parquet(ARTICLE_TABLE_PATH, columns=["article_id"])
    texts = pd.read_parquet(ARTICLES_PATH).set_index("article_id").reindex(art["article_id"])
    text = (texts["headline"].fillna("") + "\n" + texts["article_text"].fillna("").str.slice(0, 3000)).to_numpy()
    tok = AutoTokenizer.from_pretrained(config.FINETUNE_MODEL)
    rows = np.concatenate([tr, va, te])
    ids = dict(zip(rows, tok(list(text[rows]), truncation=True, max_length=config.FINETUNE_MAX_TOKENS)["input_ids"]))
    print(f"tokenized ({time.time() - t0:.0f}s)", flush=True)

    # Per-stock class weights from training labels, so "flat" doesn't drown out the moves
    counts = np.zeros((n_s, 3))
    for i in range(0, len(tr), 5000):
        y = np.asarray(y_main[tr[i:i + 5000]])
        if named is not None:
            y = np.where(named[tr[i:i + 5000]], y, -1)
        for k in range(3):
            counts[:, k] += (y == k).sum(0)
    w_main = torch.tensor(np.clip(counts.sum(1, keepdims=True) / (3 * np.maximum(counts, 1)), 0.2, 10), dtype=torch.float32, device="cuda")

    model = NewsModel(config.FINETUNE_MODEL, n_s, 2 * len(meta["price_names"]), article.shape[1]).cuda()
    enc = list(model.encoder.parameters())
    rest = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": config.FINETUNE_LR}, {"params": rest, "lr": 1e-3}], weight_decay=0.01)
    steps = args.epochs * math.ceil(len(tr) / config.NEWS_MODEL_BATCH)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)

    def batch(b):
        L = max(len(ids[i]) for i in b)
        x = np.full((len(b), L), tok.pad_token_id, np.int64)
        m = np.zeros((len(b), L), np.int64)
        for j, i in enumerate(b):
            x[j, :len(ids[i])] = ids[i]
            m[j, :len(ids[i])] = 1
        bs = np.sort(b)  # sorted rows read faster from the on-disk arrays
        order = np.searchsorted(bs, b)
        P = inputs.prices(np.asarray(prices[bs]))[order]
        A = inputs.article(article[b])
        cuda = lambda v, dt=None: torch.from_numpy(v).cuda()
        ym = np.asarray(y_main[bs])[order].astype(np.int64)
        ya = np.asarray(y_aux[bs])[order].astype(np.int64)
        if named is not None:
            ym = np.where(named[b], ym, -1)
            ya = np.where(named[b], ya, -1)
        return (cuda(x), cuda(m), cuda(A), cuda(P), torch.from_numpy(ym).cuda(), torch.from_numpy(ya).cuda(),
                cuda(art_weight[b].astype(np.float32)))

    def loss_fn(logits_main, logits_aux, ym, ya, aw):
        lm = logits_main.float().reshape(-1, 3)
        la = logits_aux.float().reshape(-1, 3)
        ym, ya = ym.reshape(-1), ya.reshape(-1)
        stock = torch.arange(n_s, device="cuda").repeat(logits_main.shape[0])
        cell_w = aw.repeat_interleave(n_s)  # the article's weight, for each of its stocks
        okm, oka = ym >= 0, ya >= 0
        wm = w_main[stock[okm], ym[okm]] * cell_w[okm]
        wa = cell_w[oka]
        if wm.sum() <= 0:  # a batch with no labelled cells (possible with --named-only/--clean-only)
            zero = lm.sum() * 0.0
            return zero, zero
        main = (nn.functional.cross_entropy(lm[okm], ym[okm], reduction="none") * wm).sum() / wm.sum()
        aux = ((nn.functional.cross_entropy(la[oka], ya[oka], reduction="none") * wa).sum() / wa.sum()) if wa.sum() > 0 else main * 0.0
        return main + AUX_WEIGHT * aux, main

    @torch.no_grad()
    def predict(rows_):
        model.eval()
        out, total, n = np.zeros((len(rows_), n_s, 3), np.float16), 0.0, 0
        for i in range(0, len(rows_), 64):
            b = rows_[i:i + 64]
            x, m, A, P, ym, ya, aw = batch(b)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lm, la = model(x, m, A, P)
            _, main = loss_fn(lm, la, ym, ya, aw)
            if bool((ym >= 0).any()):  # only batches with labels count toward the validation loss
                total += float(main) * len(b); n += len(b)
            out[i:i + len(b)] = torch.softmax(lm.float(), -1).cpu().numpy().astype(np.float16)
        model.train()
        return out, total / max(n, 1)

    best = float("inf")
    with keep_awake():
        model.train()
        step = 0
        for epoch in range(args.epochs):
            seen, run = 0, 0.0
            order = rng.permutation(tr)
            for i in range(0, len(order), config.NEWS_MODEL_BATCH):
                b = order[i:i + config.NEWS_MODEL_BATCH]
                x, m, A, P, ym, ya, aw = batch(b)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lm, la = model(x, m, A, P)
                loss, main = loss_fn(lm, la, ym, ya, aw)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1; seen += len(b); run += float(main.detach()) * len(b)
                if step % 200 == 0:
                    print(f"epoch {epoch + 1}  {seen:,}/{len(order):,} articles  loss {run / seen:.4f}  ({(time.time() - t0) / 60:.0f} min)", flush=True)
            _, val_loss = predict(va)
            print(f"epoch {epoch + 1} done: training loss {run / seen:.4f}, validation loss {val_loss:.4f} ({(time.time() - t0) / 60:.0f} min)", flush=True)
            if val_loss < best:
                best = val_loss
                torch.save({"state": model.state_dict(), "inputs": inputs.__dict__, "stocks": meta["stocks"], "epoch": epoch + 1,
                            "base": config.FINETUNE_MODEL, "n_price": 2 * len(meta["price_names"]), "n_article": article.shape[1]}, model_path)
                print(f"  saved (best so far) to {model_path}", flush=True)
        model.load_state_dict(torch.load(model_path, weights_only=False)["state"])
        probs, _ = predict(np.concatenate([va, te]))
    kind = "_named" if args.named_only else "_clean" if args.clean_only else ""
    np.save(grid / f"proba_val_test{kind}{frac_tag}{top_tag}.npy", probs)
    (grid / "proba_rows.json").write_text(json.dumps({"val": [int(va[0]), int(va[-1]) + 1], "test": [int(te[0]), int(te[-1]) + 1]}))
    print(f"Saved validation + test predictions ({probs.shape}) ({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    main()
