"""
Train a local ticker tagger that copies Claude's tags (Job 2), so new articles
can be tagged instantly and for free, without calling Claude.

Claude is the teacher: its tags in news_tagged.json are the answers. For each
company tagged in at least TAGGER_MIN_ARTICLES training articles:
  - a yes/no model learns "does this article affect this company?" from the
    article's words (TF-IDF + logistic regression, like the price model)
  - a small role model learns which role (direct, competitor, ...) the company
    plays, from the same words
Companies tagged less often can't be learned this way and are never returned.

Checked like the price model, by date: trained on the oldest 70% of articles,
the yes/no cutoff picked on the next 15%, and the newest 15% compared against
Claude's tags. Then trained again on all articles for live use and saved to
models/tagger.joblib.

Run:
    python -m newsmodel.modeling.tagger
Use:
    from newsmodel.modeling import tagger
    model = tagger.load()
    tagger.tag(model, title, body)  # -> [(ticker, role, probability), ...] most likely first
"""

import json
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from newsmodel import config, market_time

TAGGER_PATH = config.MODELS_DIR / "tagger.joblib"
ROLE_FEATURES = 5000     # the role models use only the most common words, to stay small
MIN_ROLE_EXAMPLES = 10   # rarer roles for a company are folded into its usual role


def load_articles() -> pd.DataFrame:
    """Every article with a usable news time, oldest first: text and Claude's tags."""
    rows = []
    for item in json.loads(config.TAGGED_NEWS_PATH.read_text()):
        when = market_time.news_time(item)
        if when is None:
            continue
        rows.append({
            "time": when,
            "text": f"{item.get('title') or ''}. {item.get('body') or ''}",
            "tickers": item.get("tickers") or [],
            "roles": item.get("ticker_roles") or {},
        })
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=config.TAGGER_MAX_FEATURES,
        ngram_range=(1, 2),   # "jensen huang", "crude oil"
        min_df=5,
        stop_words="english",
        sublinear_tf=True,    # a name said 10 times isn't 10x the evidence of saying it once
        dtype=np.float32,
    )


def common_tickers(df: pd.DataFrame) -> list:
    counts = Counter(t for tickers in df["tickers"] for t in tickers)
    return sorted(t for t, n in counts.items() if n >= config.TAGGER_MIN_ARTICLES)


def label_matrix(df: pd.DataFrame, tickers: list) -> np.ndarray:
    """Y[i, j] = Claude tagged article i with tickers[j]."""
    column = {t: j for j, t in enumerate(tickers)}
    Y = np.zeros((len(df), len(tickers)), dtype=bool)
    for i, tagged in enumerate(df["tickers"]):
        for t in tagged:
            if t in column:
                Y[i, column[t]] = True
    return Y


def fit_yes_no(X, y):
    model = LogisticRegression(solver="liblinear", C=config.TAGGER_C)
    model.fit(X, y)
    return model.coef_[0].astype(np.float32), np.float32(model.intercept_[0])


def fit_roles(X_role, roles: list):
    """A role model for one company, or just its usual role if it rarely plays another."""
    counts = Counter(roles)
    usual = counts.most_common(1)[0][0]
    learnable = {r for r, n in counts.items() if n >= MIN_ROLE_EXAMPLES}
    if len(learnable) < 2:
        return usual
    model = LogisticRegression(max_iter=1000)
    model.fit(X_role, [r if r in learnable else usual for r in roles])
    return model


def fit(df: pd.DataFrame) -> dict:
    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(df["text"])
    X.sort_indices()  # the parallel workers get a read-only copy and can't sort it themselves
    tickers = common_tickers(df)
    Y = label_matrix(df, tickers)
    print(f"  {len(df)} articles, {len(tickers)} companies, {X.shape[1]} words/phrases", flush=True)

    results = Parallel(n_jobs=config.TAGGER_JOBS)(delayed(fit_yes_no)(X, Y[:, j]) for j in range(len(tickers)))
    W = np.vstack([w for w, _ in results])
    b = np.array([b for _, b in results], dtype=np.float32)

    doc_freq = np.asarray((X > 0).sum(axis=0)).ravel()
    role_cols = np.sort(np.argsort(-doc_freq)[:ROLE_FEATURES])
    X_role = X[:, role_cols]
    role_models = {}
    for j, ticker in enumerate(tickers):
        rows = np.flatnonzero(Y[:, j])
        roles = [df.at[i, "roles"].get(ticker, "other") for i in rows]
        role_models[ticker] = fit_roles(X_role[rows], roles)

    return {"vectorizer": vectorizer, "tickers": tickers, "W": W, "b": b,
            "role_cols": role_cols, "role_models": role_models, "threshold": 0.5}


def probabilities(model: dict, X) -> np.ndarray:
    """P[i, j] = chance article i affects model["tickers"][j]."""
    z = np.asarray(X @ model["W"].T) + model["b"]
    return 1 / (1 + np.exp(-z))


def pick(p_row: np.ndarray, threshold: float) -> np.ndarray:
    """Columns at or above the cutoff, most likely first, at most MAX_COMPANIES_PER_ARTICLE."""
    found = np.flatnonzero(p_row >= threshold)
    return found[np.argsort(-p_row[found])][:config.MAX_COMPANIES_PER_ARTICLE]


def predict_role(model: dict, x_role, ticker: str) -> str:
    role_model = model["role_models"][ticker]
    return role_model if isinstance(role_model, str) else role_model.predict(x_role)[0]


def micro_scores(P: np.ndarray, Y: np.ndarray, threshold: float):
    hits = found = 0
    for p_row, y_row in zip(P, Y):
        chosen = pick(p_row, threshold)
        found += len(chosen)
        hits += y_row[chosen].sum()
    precision = hits / found if found else 0.0
    recall = hits / Y.sum() if Y.sum() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate(model: dict, df: pd.DataFrame, name: str):
    """Compare the tagger with Claude's tags on articles it wasn't trained on."""
    tickers = model["tickers"]
    X = model["vectorizer"].transform(df["text"])
    P = probabilities(model, X)
    Y = label_matrix(df, tickers)
    threshold = model["threshold"]
    precision, recall, f1 = micro_scores(P, Y, threshold)

    all_claude = sum(len(t) for t in df["tickers"])
    print(f"\n--- {name}: {len(df)} articles ({df['time'].min().date()} to {df['time'].max().date()}) ---")
    print(f"Cutoff: {threshold:.2f}")
    print(f"Precision: {precision:.3f}  (of the companies it found, share Claude also tagged)")
    print(f"Recall:    {recall:.3f}  (of Claude's tags among the {len(tickers)} learned companies, share it found)")
    print(f"Recall of ALL Claude tags, incl. companies it can't learn: {recall * Y.sum() / all_claude:.3f}")

    # Top pick vs. Claude's first (most affected) company, where that one is learnable
    column = {t: j for j, t in enumerate(tickers)}
    firsts = [(i, column[t[0]]) for i, t in enumerate(df["tickers"]) if t and t[0] in column]
    top1 = np.mean([P[i].argmax() == j for i, j in firsts])
    print(f"Its most likely company = Claude's most affected one: {top1:.3f} ({len(firsts)} articles)")

    # Roles, on companies both found
    X_role = X[:, model["role_cols"]]
    right = total = 0
    for i in range(len(df)):
        for j in pick(P[i], threshold):
            if Y[i, j]:
                total += 1
                right += predict_role(model, X_role[i], tickers[j]) == df.at[i, "roles"].get(tickers[j])
    print(f"Role matches Claude's, where both found the company: {right / total:.3f} ({total} pairs)")

    # Per company
    rows = []
    for j, ticker in enumerate(tickers):
        n_claude = Y[:, j].sum()
        chosen = P[:, j] >= threshold
        hits = (chosen & Y[:, j]).sum()
        rows.append({"ticker": ticker, "claude": n_claude, "found": chosen.sum(),
                     "precision": hits / chosen.sum() if chosen.sum() else np.nan,
                     "recall": hits / n_claude if n_claude else np.nan})
    per = pd.DataFrame(rows).set_index("ticker")
    print("\nMost-tagged companies:")
    print(per.sort_values("claude", ascending=False).head(12).round(2).to_string())
    worst = per[per["claude"] >= 20].assign(f1=lambda d: 2 * d.precision * d.recall / (d.precision + d.recall))
    print("\nWorst-learned companies (20+ Claude tags here):")
    print(worst.sort_values("f1").head(10).round(2).to_string())


def tag(model: dict, title: str, body: str) -> list:
    """[(ticker, role, probability), ...] for one new article, most likely first."""
    X = model["vectorizer"].transform([f"{title}. {body}"])
    p_row = probabilities(model, X)[0]
    X_role = X[:, model["role_cols"]]
    return [(model["tickers"][j], predict_role(model, X_role, model["tickers"][j]), float(p_row[j]))
            for j in pick(p_row, model["threshold"])]


def load() -> dict:
    if not TAGGER_PATH.exists():
        raise SystemExit("No tagger yet. Run python -m newsmodel.modeling.tagger first.")
    return joblib.load(TAGGER_PATH)


def main():
    if not config.TAGGED_NEWS_PATH.exists():
        raise SystemExit("No tagged news found. Run Jobs 1 and 2 first.")
    df = load_articles()
    train_end = int(len(df) * config.TRAIN_FRACTION)
    val_end = int(len(df) * (config.TRAIN_FRACTION + config.VAL_FRACTION))
    train, val, test = df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]
    val, test = val.reset_index(drop=True), test.reset_index(drop=True)

    print("Training on the oldest articles to check the tagger against Claude...", flush=True)
    model = fit(train)

    # The cutoff that best matches Claude on validation (never on test)
    P_val = probabilities(model, model["vectorizer"].transform(val["text"]))
    Y_val = label_matrix(val, model["tickers"])
    best = max(np.arange(0.1, 0.91, 0.05), key=lambda t: micro_scores(P_val, Y_val, t)[2])
    model["threshold"] = float(best)
    evaluate(model, test, "TEST vs. Claude (the number that matters)")

    print("\nTraining again on all articles for live use...", flush=True)
    final = fit(df)
    final["threshold"] = float(best)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, TAGGER_PATH)
    print(f"Saved tagger ({len(final['tickers'])} companies) to {TAGGER_PATH}")


if __name__ == "__main__":
    main()
