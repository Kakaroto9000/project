"""
Train a classifier to predict up/down/flat from the news text plus when the
news came out. Second step of Job 3.

Inputs:
  - TF-IDF over headline + article text
  - news_session (weekend / holiday / pre_market / market_hours /
    after_hours), one-hot encoded, so the model can treat news that waits a
    weekend to be priced in differently from news during trading
  - role (direct / competitor / opposite_market / supplier_customer /
    world_event / policy_macro / other), one-hot encoded: one article is one row per company, so
    without it the model sees the same text with opposite labels when a
    company's win sinks its competitor

No sentiment_score (or any other source-provided signal) goes into the
model: this answers "does the news itself predict price direction better
than always guessing the most common label?"

The model only ever outputs one of the three labels, never a percentage.
If "flat" is near 0% or near 100% of rows in the label distribution
modeling/dataset.py prints, revisit FLAT_THRESHOLD_PCT.

Run (normally via job3-train):
    python -m newsmodel.modeling.train
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import joblib

from newsmodel import config
from newsmodel.modeling.dataset import DATASET_PATH

MODEL_PATH = config.MODELS_DIR / "baseline_model.joblib"
TEXT_COL = "combined_text"  # built from "headline" + "article_text" below
SESSION_COL = "news_session"
ROLE_COL = "role"


def chronological_split(df: pd.DataFrame):
    """
    Sort by time and split sequentially -- NEVER shuffle randomly. A random
    shuffle would let the model train on news that happened AFTER some of
    its test examples, silently inflating test accuracy with information
    it wouldn't have in live trading.
    """
    df = df.sort_values("news_timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * config.TRAIN_FRACTION)
    val_end = int(n * (config.TRAIN_FRACTION + config.VAL_FRACTION))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    print(f"Train: {len(train)} rows ({train['news_timestamp'].min()} to {train['news_timestamp'].max()})")
    print(f"Val:   {len(val)} rows ({val['news_timestamp'].min()} to {val['news_timestamp'].max()})")
    print(f"Test:  {len(test)} rows ({test['news_timestamp'].min()} to {test['news_timestamp'].max()})")

    return train, val, test


def add_combined_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TEXT_COL] = (df["headline"].fillna("") + ". " + df["article_text"].fillna("")).str.strip()
    return df


def build_featurizer() -> ColumnTransformer:
    return ColumnTransformer([
        ("text", TfidfVectorizer(
            max_features=config.TEXT_MAX_FEATURES,
            ngram_range=(1, 2),
            min_df=5,
            stop_words="english",
        ), TEXT_COL),
        ("session", OneHotEncoder(handle_unknown="ignore"), [SESSION_COL]),
        ("role", OneHotEncoder(handle_unknown="ignore"), [ROLE_COL]),
    ])


def build_features(df: pd.DataFrame, featurizer: ColumnTransformer, fit: bool):
    df = add_combined_text(df)
    return featurizer.fit_transform(df) if fit else featurizer.transform(df)


def evaluate(model, X, y, split_name: str):
    preds = model.predict(X)
    print(f"\n--- {split_name} ---")
    print(classification_report(y, preds, zero_division=0))
    print("Confusion matrix (rows=actual, cols=predicted), labels=[down, flat, up]:")
    print(confusion_matrix(y, preds, labels=["down", "flat", "up"]))


def main():
    if not DATASET_PATH.exists():
        raise SystemExit("No dataset yet. Run python -m newsmodel.modeling.dataset (or job3-train) first.")

    df = pd.read_parquet(DATASET_PATH)
    train, val, test = chronological_split(df)

    featurizer = build_featurizer()
    X_train = build_features(train, featurizer, fit=True)
    X_val = build_features(val, featurizer, fit=False)
    X_test = build_features(test, featurizer, fit=False)

    y_train, y_val, y_test = train["label"], val["label"], test["label"]

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    evaluate(model, X_train, y_train, "TRAIN (sanity check only -- expect this to look better than test)")
    evaluate(model, X_val, y_val, "VALIDATION")
    evaluate(model, X_test, y_test, "TEST (the number that actually matters)")

    # --- The single most important sanity check in this whole pipeline ---
    majority_class = y_train.mode()[0]
    baseline_acc = (y_test == majority_class).mean()
    print(f"\nNaive baseline (always predict '{majority_class}'): {baseline_acc:.3f} accuracy on test")
    print("If test accuracy isn't clearly above this, the news is not predicting direction")
    print("in this window -- a real, useful finding, not a failure of the code.")

    # Show what the model learned: the highest-weighted words per class, and
    # how each news timing and each role pushes toward each class.
    feature_names = featurizer.get_feature_names_out()
    text_idx = [i for i, name in enumerate(feature_names) if name.startswith("text__")]
    top_n = 10
    print(f"\nTop {top_n} words pushing toward each class:")
    for cls, coef in zip(model.classes_, model.coef_):
        top = sorted(text_idx, key=lambda i: coef[i], reverse=True)[:top_n]
        print(f"  {cls}: {', '.join(feature_names[i].removeprefix('text__') for i in top)}")
    for title, prefix in [("News timing", f"session__{SESSION_COL}_"), ("Role", f"role__{ROLE_COL}_")]:
        print(f"\n{title} weight per class (positive = pushes toward that class):")
        for i, name in enumerate(feature_names):
            if name.startswith(prefix):
                weights = ", ".join(f"{cls} {coef[i]:+.2f}" for cls, coef in zip(model.classes_, model.coef_))
                print(f"  {name.removeprefix(prefix):17s} {weights}")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "featurizer": featurizer}, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
