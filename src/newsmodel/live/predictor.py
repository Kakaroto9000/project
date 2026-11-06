"""
The saved one-model-for-all-stocks (modeling/news_model.py) used on one new
article at a time, for the stocks the app watches: config.INDEX_TICKERS plus
the LIVE_TOP_COMPANIES most traded companies (universe.json is sorted by dollar
volume; funds like VOO or GLD are left out of the company count).

Each stock's answer only depends on the article and that stock's own prices,
so stocks outside the watchlist are simply fed "missing" prices and ignored.
"""

import json

import numpy as np
import torch

from newsmodel import config
from newsmodel.collect.universe import UNIVERSE_PATH
from newsmodel.modeling.news_model import Inputs, NewsModel
from newsmodel.modeling.universe_dataset import FUND_WORDS


def watchlist(model_stocks: list, top: int = None) -> list:
    """INDEX_TICKERS, then the `top` (LIVE_TOP_COMPANIES) most traded companies the model knows."""
    known = set(model_stocks)
    companies = [u["ticker"] for u in json.loads(UNIVERSE_PATH.read_text())
                 if not FUND_WORDS.search(u["name"] or "") and u["ticker"] not in config.INDEX_TICKERS and u["ticker"] in known]
    return [t for t in config.INDEX_TICKERS if t in known] + companies[:top or config.LIVE_TOP_COMPANIES]


def article_text(title: str, body: str) -> str:
    return (title or "") + "\n" + (body or "")[:3000]  # as news_model.py's training text


class Predictor:
    def __init__(self, path=None):
        from transformers import AutoTokenizer
        path = path or config.LIVE_MODEL_PATH
        ck = torch.load(path, weights_only=False, map_location="cpu")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = NewsModel(ck["base"], len(ck["stocks"]), ck["n_price"], ck["n_article"])
        self.model.load_state_dict(ck["state"])
        self.model.to(self.device).eval()
        self.inputs = Inputs()
        self.inputs.__dict__.update(ck["inputs"])
        self.tok = AutoTokenizer.from_pretrained(ck["base"])
        self.stocks = watchlist(ck["stocks"])
        index = {t: i for i, t in enumerate(ck["stocks"])}
        self.cols = np.array([index[t] for t in self.stocks])
        self.n_model_stocks = len(ck["stocks"])
        self.name = path.name

    @torch.no_grad()
    def predict(self, text: str, prices: np.ndarray, article: np.ndarray) -> np.ndarray:
        """(len(self.stocks), 3) probabilities of down / flat / up. prices: (len(self.stocks), 6)."""
        enc = self.tok([text], truncation=True, max_length=config.FINETUNE_MAX_TOKENS, return_tensors="pt")
        full = np.full((self.n_model_stocks, prices.shape[1]), np.nan, np.float32)
        full[self.cols] = prices
        P = torch.from_numpy(self.inputs.prices(full)[None]).to(self.device)
        A = torch.from_numpy(self.inputs.article(article[None].astype(np.float32))).to(self.device)
        with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
            logits, _ = self.model(enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device), A, P)
        return torch.softmax(logits.float(), -1)[0, self.cols].cpu().numpy()
