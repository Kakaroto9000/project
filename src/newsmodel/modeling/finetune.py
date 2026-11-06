"""
Fine-tune a pretrained language model (FINETUNE_MODEL, ModernBERT-base) on our own
labels, so it learns which news moves prices -- not just what the words mean. Every
layer is trained, on the GPU.

Two modes, on modeling/universe_dataset.py's pairs, split by article time like
the other models (oldest TRAIN_FRACTION trains, then validation, then test):

  company  one example per (article, company named in it): the model reads
           "<company name> (<TICKER>), named in the headline" next to the
           article and predicts that company's next-30-minute move vs. SPY
           (up / down / flat, as in universe_dataset.py).
  funds    one example per article; the model predicts all INDEX_TICKERS funds
           at once (one up/down/flat answer per fund).

With --prices, the price inputs (moves before the news, volume, trend,
volatility, opening gap, ...) go into the model's final layer next to the text,
so it can learn "good news, but the stock already jumped".

Saves the best epoch (lowest validation loss) to models/finetune_<name>.pt and
its probabilities for every validation and test example to
data/processed/finetune_<name>_proba.parquet.

Run:
    python -m newsmodel.modeling.finetune company --name f1_company
    python -m newsmodel.modeling.finetune company --prices --name f2_company_prices
    python -m newsmodel.modeling.finetune funds --prices --name f3_funds_prices
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
from newsmodel.collect.universe import UNIVERSE_PATH
from newsmodel.keep_awake import keep_awake
from newsmodel.modeling.dataset import ARTICLES_PATH
from newsmodel.modeling.universe_dataset import ARTICLE_TABLE_PATH, PAIRS_PATH

LABELS = ["down", "flat", "up"]
PRICE_COLS = ["pre_news_pct", "open_gap_pct", "volume_ratio", "ret_1d_pct", "ret_5d_pct", "volatility_20d_pct",
              "beta", "log_dollar_volume", "peer_max", "tagger_p", "named_count", "n_named"]
ARTICLE_PRICE_COLS = ["spy_pre_news_pct", "spy_open_gap_pct", "hours_to_entry", "label_minutes"]


def load_examples(mode: str):
    ds = pd.read_parquet(PAIRS_PATH)
    art = pd.read_parquet(ARTICLE_TABLE_PATH)
    n = len(art)
    cut1, cut2 = int(n * config.TRAIN_FRACTION), int(n * (config.TRAIN_FRACTION + config.VAL_FRACTION))
    split = np.where(np.arange(n) < cut1, "train", np.where(np.arange(n) < cut2, "val", "test"))
    texts = pd.read_parquet(ARTICLES_PATH).set_index("article_id").reindex(art["article_id"])
    article_text = (texts["headline"].fillna("") + "\n" + texts["article_text"].fillna("").str.slice(0, 3000)).to_numpy()
    names = {u["ticker"]: u["name"] for u in json.loads(UNIVERSE_PATH.read_text())}
    for c in ARTICLE_PRICE_COLS:
        ds[c] = art[c].to_numpy()[ds["article_idx"]]

    if mode == "company":
        ex = ds[(ds["named_count"] > 0) & ~ds["is_fund"]].reset_index(drop=True)
        head = np.where(ex["named_in_headline"], "named in the headline", "named in the article")
        ex["prompt"] = [f"{names.get(t, t)} ({t}), {h}" for t, h in zip(ex["ticker"], head)]
        ex["text"] = article_text[ex["article_idx"]]
        ex["y"] = ex["label"].map({k: i for i, k in enumerate(LABELS)})
        ex["split"] = split[ex["article_idx"]]
        return ex, PRICE_COLS + ARTICLE_PRICE_COLS, 1

    funds = [t for t in config.INDEX_TICKERS]
    f = ds[ds["ticker"].isin(funds)]
    wide_y = f.pivot_table(index="article_idx", columns="ticker", values="label", aggfunc="first").reindex(columns=funds)
    ex = pd.DataFrame({"article_idx": wide_y.index.to_numpy()})
    for t in funds:
        ex[f"y_{t}"] = wide_y[t].map({k: i for i, k in enumerate(LABELS)}).fillna(-100).astype(int).to_numpy()
    # Per-article price inputs: SPY's own pre-news numbers plus each fund's own move before the news
    for c in ARTICLE_PRICE_COLS:
        ex[c] = art[c].to_numpy()[ex["article_idx"]]
    own = f.pivot_table(index="article_idx", columns="ticker", values=["pre_news_pct", "open_gap_pct", "ret_1d_pct", "ret_5d_pct", "volatility_20d_pct"])
    own.columns = [f"{a}_{b}" for a, b in own.columns]
    ex = ex.join(own, on="article_idx")
    ex["prompt"] = "Market and sector funds: " + ", ".join(funds)
    ex["text"] = article_text[ex["article_idx"]]
    ex["split"] = split[ex["article_idx"]]
    return ex, ARTICLE_PRICE_COLS + list(own.columns), len(funds)


class PriceScaler:
    """Median-fill and rank-map to a bell curve, fitted on training rows (like train.py)."""

    def fit(self, X: np.ndarray):
        from sklearn.preprocessing import QuantileTransformer
        self.median = np.nanmedian(X, axis=0)
        self.qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000, subsample=200_000, random_state=0)
        self.qt.fit(np.where(np.isnan(X), self.median, X))
        return self

    def transform(self, X):
        return self.qt.transform(np.where(np.isnan(X), self.median, X)).astype(np.float32)


class NewsModel(nn.Module):
    def __init__(self, base: str, n_prices: int, n_outputs: int):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(base)
        hidden = self.encoder.config.hidden_size
        self.prices = nn.Sequential(nn.Linear(n_prices, 64), nn.GELU()) if n_prices else None
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden + (64 if n_prices else 0), 256), nn.GELU(),
                                  nn.Dropout(0.1), nn.Linear(256, 3 * n_outputs))
        self.n_outputs = n_outputs

    def forward(self, input_ids, attention_mask, prices=None):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = (h * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)  # mean over tokens
        if self.prices is not None:
            pooled = torch.cat([pooled, self.prices(prices)], dim=1)
        return self.head(pooled).view(-1, self.n_outputs, 3)


def batches(idx, size, shuffle, rng):
    idx = rng.permutation(idx) if shuffle else idx
    for i in range(0, len(idx), size):
        yield idx[i:i + size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["company", "funds"])
    ap.add_argument("--prices", action="store_true")
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=config.FINETUNE_EPOCHS)
    ap.add_argument("--limit", type=int, default=0, help="train on this many examples only (quick test)")
    args = ap.parse_args()
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    t0 = time.time()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    ex, price_cols, n_out = load_examples(args.mode)
    y_cols = ["y"] if args.mode == "company" else [c for c in ex.columns if c.startswith("y_")]
    Y = ex[y_cols].to_numpy(np.int64)
    tr = np.flatnonzero(ex["split"].to_numpy() == "train")
    va = np.flatnonzero(ex["split"].to_numpy() == "val")
    te = np.flatnonzero(ex["split"].to_numpy() == "test")
    if args.limit:
        tr, va, te = tr[:args.limit], va[:args.limit // 5], te[:args.limit // 5]
    print(f"[{args.name}] {len(tr)} train / {len(va)} val / {len(te)} test examples, {n_out} output(s), "
          f"prices {'on' if args.prices else 'off'} ({time.time() - t0:.0f}s)", flush=True)

    P = None
    if args.prices:
        scaler = PriceScaler().fit(ex[price_cols].to_numpy(np.float64)[tr])
        P = scaler.transform(ex[price_cols].to_numpy(np.float64))
    tok = AutoTokenizer.from_pretrained(config.FINETUNE_MODEL)
    # Each article and each prompt is tokenized once, then joined per example:
    # [CLS] prompt [SEP] article (cut to fit) [SEP]
    used = np.unique(np.concatenate([tr, va, te]))
    art_ids = ex["article_idx"].to_numpy()
    uniq_art = np.unique(art_ids[used])
    text_of = dict(zip(art_ids, ex["text"].to_numpy()))
    art_tok = dict(zip(uniq_art, tok([text_of[a] for a in uniq_art], add_special_tokens=False, truncation=True,
                                     max_length=config.FINETUNE_MAX_TOKENS)["input_ids"]))
    prompts = ex["prompt"].to_numpy()
    uniq_p = np.unique(prompts[used])
    p_tok = dict(zip(uniq_p, tok(list(uniq_p), add_special_tokens=False)["input_ids"]))
    cls, sep = tok.cls_token_id, tok.sep_token_id
    ids = {}
    for i in used:
        p = p_tok[prompts[i]]
        ids[i] = [cls] + p + [sep] + art_tok[art_ids[i]][:config.FINETUNE_MAX_TOKENS - len(p) - 3] + [sep]
    print(f"[{args.name}] tokenized {len(uniq_art)} articles ({time.time() - t0:.0f}s)", flush=True)

    # Balanced class weights, from training labels, per output
    counts = np.stack([(Y[tr] == k).sum(0) for k in range(3)], axis=1).astype(np.float64)  # (n_out, 3)
    weights = torch.tensor(counts.sum(1, keepdims=True) / (3 * np.maximum(counts, 1)), dtype=torch.float32, device="cuda")

    model = NewsModel(config.FINETUNE_MODEL, len(price_cols) if args.prices else 0, n_out).cuda()
    enc_params = list(model.encoder.parameters())
    other = [p for n_, p in model.named_parameters() if not n_.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": config.FINETUNE_LR}, {"params": other, "lr": 1e-3}], weight_decay=0.01)
    steps = args.epochs * math.ceil(len(tr) / config.FINETUNE_BATCH)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)

    def collate(rows):
        L = max(len(ids[i]) for i in rows)
        x = np.full((len(rows), L), tok.pad_token_id, np.int64)
        m = np.zeros((len(rows), L), np.int64)
        for j, i in enumerate(rows):
            x[j, :len(ids[i])] = ids[i]
            m[j, :len(ids[i])] = 1
        out = [torch.from_numpy(x).cuda(), torch.from_numpy(m).cuda()]
        out.append(torch.from_numpy(P[rows]).cuda() if P is not None else None)
        return out

    def loss_fn(logits, y):
        total, n = 0.0, 0
        for k in range(n_out):
            ok = y[:, k] >= 0
            if ok.any():
                total = total + nn.functional.cross_entropy(logits[ok, k].float(), y[ok, k], weight=weights[k], reduction="sum")
                n += int(ok.sum())
        return total / max(n, 1)

    @torch.no_grad()
    def predict(rows):
        model.eval()
        probs, losses = [], []
        order = rows[np.argsort([len(ids[i]) for i in rows])]  # similar lengths together: faster
        for b in batches(order, 128, False, rng):
            x, m, p = collate(b)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x, m, p)
            losses.append(float(loss_fn(logits, torch.from_numpy(Y[b]).cuda())) * len(b))
            probs.append(torch.softmax(logits.float(), -1).cpu().numpy())
        model.train()
        pr = np.concatenate(probs)  # in `order`; put back in `rows` order
        result = np.empty_like(pr)
        pos = {r: i for i, r in enumerate(rows)}
        result[[pos[r] for r in order]] = pr
        return result, sum(losses) / len(rows)

    best, best_loss = None, float("inf")
    path = config.MODELS_DIR / f"finetune_{args.name}.pt"
    with keep_awake():
        model.train()
        step = 0
        for epoch in range(args.epochs):
            seen, run_loss = 0, 0.0
            for b in batches(tr, config.FINETUNE_BATCH, True, rng):
                x, m, p = collate(b)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x, m, p)
                loss = loss_fn(logits, torch.from_numpy(Y[b]).cuda())
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1; seen += len(b); run_loss += float(loss) * len(b)
                if step % 500 == 0:
                    print(f"[{args.name}] epoch {epoch + 1} {seen}/{len(tr)} loss {run_loss / seen:.4f} "
                          f"({(time.time() - t0) / 60:.0f} min)", flush=True)
            _, val_loss = predict(va)
            print(f"[{args.name}] epoch {epoch + 1} done: train loss {run_loss / seen:.4f}, validation loss {val_loss:.4f} "
                  f"({(time.time() - t0) / 60:.0f} min)", flush=True)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({"state": model.state_dict(), "price_cols": price_cols if args.prices else [], "mode": args.mode,
                            "n_out": n_out, "base": config.FINETUNE_MODEL}, path)

        model.load_state_dict(torch.load(path)["state"])
        rows = np.concatenate([va, te])
        probs, _ = predict(rows)
    out = ex.iloc[rows][["article_idx"] + (["company_idx", "ticker"] if args.mode == "company" else [])].reset_index(drop=True)
    out["split"] = ex["split"].to_numpy()[rows]
    if args.mode == "company":
        for k, lab in enumerate(LABELS):
            out[f"p_{lab}"] = probs[:, 0, k]
    else:
        for j, t in enumerate(config.INDEX_TICKERS):
            for k, lab in enumerate(LABELS):
                out[f"p_{t}_{lab}"] = probs[:, j, k]
    dest = config.PROCESSED_DIR / f"finetune_{args.name}_proba.parquet"
    out.to_parquet(dest, index=False)
    print(f"[{args.name}] saved model to {path} and {len(out)} predictions to {dest} ({(time.time() - t0) / 60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
