"""
Train a classifier to predict up/down/flat over the HORIZON_MINUTES after
the news, from the news text, when it came out, and how the stock and the
market moved before it. Second step of Job 3.

The move starts at the news, or at the next open if the market is closed.
Inputs, all known when it starts (modeling/dataset.py builds them):
  - TF-IDF over headline + article text
  - price moves before it: the stock's over the PRE_NEWS_MINUTES before the
    news (pre_news_pct) and the market's (spy_pre_news_pct), the jump from
    the previous close to that day's open (open_gap_pct, spy_open_gap_pct),
    trading volume before the news vs. usual (volume_ratio, market hours
    only), the trend over the days before (ret_*d_pct) and the stock's
    typical daily move (volatility_*d_pct)
  - how long the news waited before the trade (hours_to_entry), and how
    long the predicted move lasts (label_minutes: HORIZON_MINUTES, less for
    news just before the close)
  - article details: how many companies it affects (n_tickers) and this
    one's rank among them (ticker_rank)
  - categories, one-hot encoded:
      news_session (weekend / holiday / pre_market / market_hours /
      after_hours) -- news that waits a weekend to be priced in can behave
      differently from news during trading
      entry (at_news / next_open) -- whether the trade waited for the next
      open; also catches market_hours news with under LATE_NEWS_MIN_MINUTES left
      role (direct / competitor / ...) -- one article is one row per
      company, so without it the model sees the same text with opposite
      labels when a company's win sinks its competitor
      source_name (the outlet) and day_of_week

No sentiment_score (or any other source-provided signal) goes into the
model: this answers "does the news itself predict price direction better
than always guessing the most common label?"

The model only ever outputs one of the three labels, never a percentage.
If "flat" is near 0% or near 100% of rows in the label distribution
modeling/dataset.py prints, revisit FLAT_THRESHOLD_FRACTION.

Run (normally via job3-train):
    python -m newsmodel.modeling.train
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
import joblib

from newsmodel import config
from newsmodel.modeling.dataset import ARTICLES_PATH, DATASET_PATH

MODEL_PATH = config.MODELS_DIR / "baseline_model.joblib"
NUMERIC_COLS = [
    "pre_news_pct", "spy_pre_news_pct", "open_gap_pct", "spy_open_gap_pct", "volume_ratio",
    *(f"ret_{days}d_pct" for days in config.TREND_DAYS), f"volatility_{config.VOLATILITY_DAYS}d_pct",
    "hours_to_entry", "label_minutes", "n_tickers", "ticker_rank",
]
CATEGORY_COLS = ["news_session", "entry", "role", "source_name", "day_of_week"]


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


def load_articles() -> pd.Series:
    """Each article's text (headline + body), by article_id."""
    if not ARTICLES_PATH.exists():
        raise SystemExit("No article text yet. Run python -m newsmodel.modeling.dataset (or job3-train) first.")
    articles = pd.read_parquet(ARTICLES_PATH)
    text = (articles["headline"].fillna("") + ". " + articles["article_text"].fillna("")).str.strip()
    return pd.Series(text.values, index=articles["article_id"])


class Featurizer:
    """
    Turns rows into the model's inputs. The text is turned into numbers once
    per article, not once per row -- an article has a row per company and per
    fund, all with the same text -- and each row reuses its article's numbers.
    """

    def __init__(self):
        self.text = TfidfVectorizer(
            max_features=config.TEXT_MAX_FEATURES,
            ngram_range=(1, 2),
            min_df=5,
            stop_words="english",
            dtype=np.float32,
        )
        self.rest = ColumnTransformer([
            # Price moves have fat tails (a biotech can jump 300% on trial results), so
            # numbers are mapped to a bell curve by rank: one huge move can't swamp
            # the rest. Missing values (e.g. no 5-day history yet) get the median.
            ("numbers", make_pipeline(
                SimpleImputer(strategy="median"),
                QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=0),
            ), NUMERIC_COLS),
            ("categories", OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=config.MIN_CATEGORY_ROWS,
            ), CATEGORY_COLS),
        ])

    def _transform(self, df: pd.DataFrame, texts: pd.Series, fit: bool):
        codes, article_ids = pd.factorize(df["article_id"])
        unique_texts = texts.reindex(article_ids).fillna("")
        text = self.text.fit_transform(unique_texts) if fit else self.text.transform(unique_texts)
        rest = self.rest.fit_transform(df) if fit else self.rest.transform(df)
        return sp.hstack([text[codes], rest], format="csr")

    def fit_transform(self, df: pd.DataFrame, texts: pd.Series):
        return self._transform(df, texts, fit=True)

    def transform(self, df: pd.DataFrame, texts: pd.Series):
        return self._transform(df, texts, fit=False)

    def get_feature_names_out(self):
        words = [f"text__{w}" for w in self.text.get_feature_names_out()]
        return np.array(words + list(self.rest.get_feature_names_out()))


def build_featurizer() -> Featurizer:
    return Featurizer()


def build_features(df: pd.DataFrame, featurizer: Featurizer, fit: bool, texts: pd.Series):
    return featurizer.fit_transform(df, texts) if fit else featurizer.transform(df, texts)


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
    texts = load_articles()
    train, val, test = chronological_split(df)

    featurizer = build_featurizer()
    X_train = build_features(train, featurizer, fit=True, texts=texts)
    X_val = build_features(val, featurizer, fit=False, texts=texts)
    X_test = build_features(test, featurizer, fit=False, texts=texts)

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
    # how each other input pushes toward each class.
    feature_names = featurizer.get_feature_names_out()
    text_idx = [i for i, name in enumerate(feature_names) if name.startswith("text__")]
    top_n = 10
    print(f"\nTop {top_n} words pushing toward each class:")
    for cls, coef in zip(model.classes_, model.coef_):
        top = sorted(text_idx, key=lambda i: coef[i], reverse=True)[:top_n]
        print(f"  {cls}: {', '.join(feature_names[i].removeprefix('text__') for i in top)}")
    groups = [("Price and article numbers (per step up the bell curve)", "numbers__")]
    groups += [(col, f"categories__{col}_") for col in CATEGORY_COLS]
    for title, prefix in groups:
        print(f"\n{title} weight per class (positive = pushes toward that class):")
        for i, name in enumerate(feature_names):
            if name.startswith(prefix):
                weights = ", ".join(f"{cls} {coef[i]:+.2f}" for cls, coef in zip(model.classes_, model.coef_))
                print(f"  {name.removeprefix(prefix):24s} {weights}")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "featurizer": featurizer}, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
