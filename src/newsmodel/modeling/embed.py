"""
Turn every article into EMBEDDING_DIMS numbers that capture its meaning, with a
small pretrained text model (EMBEDDING_MODEL), so "beat estimates" and "topped
forecasts" land close together -- which word counts (TF-IDF) can't do.

Runs on the GPU if there is one (~10-20 min for ~170k articles on an RTX 4070),
otherwise the CPU (much slower). Reads data/processed/articles.parquet
(modeling/dataset.py) and writes data/processed/article_embeddings.parquet:
one row per article_id, columns e0..e<N-1>.

The model was trained on text from before 2024, so it can't have read how
the market later reacted to these articles.

Run (after modeling/dataset.py):
    python -m newsmodel.modeling.embed
"""

import numpy as np
import pandas as pd

from newsmodel import config
from newsmodel.keep_awake import keep_awake
from newsmodel.modeling.dataset import ARTICLES_PATH

EMBEDDINGS_PATH = config.PROCESSED_DIR / "article_embeddings.parquet"


def article_texts(articles: pd.DataFrame) -> list:
    """Headline + the start of the body: the model reads EMBEDDING_MAX_TOKENS tokens at most."""
    body = articles["article_text"].fillna("").str.slice(0, 2000)
    return (articles["headline"].fillna("") + ". " + body).tolist()


def main():
    from sentence_transformers import SentenceTransformer
    import torch

    articles = pd.read_parquet(ARTICLES_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding {len(articles)} articles with {config.EMBEDDING_MODEL} on {device}...", flush=True)
    with keep_awake():
        model = SentenceTransformer(config.EMBEDDING_MODEL, device=device)
        model.max_seq_length = config.EMBEDDING_MAX_TOKENS
        if device == "cuda":
            model.half()
        vectors = model.encode(article_texts(articles), batch_size=256, show_progress_bar=True,
                               normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    out = pd.DataFrame(vectors, columns=[f"e{i}" for i in range(vectors.shape[1])])
    out.insert(0, "article_id", articles["article_id"].values)
    out.to_parquet(EMBEDDINGS_PATH, index=False)
    print(f"Saved {vectors.shape} to {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
