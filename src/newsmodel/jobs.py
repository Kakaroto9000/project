"""
The three separately-launched jobs. Each is installed as a command by
`pip install -e .` (see pyproject.toml):

    job1-fetch-news    Event Registry -> data/raw_news/news_raw.json
    job2-tag-submit    sends news_raw.json articles to Claude (Batch API) for ticker tagging
    job2-tag-collect   once Claude is done (re-run until it is): -> news_tagged.json
    job3-train         dataset -> model -> backtest (local only, safe to re-run)
"""

from newsmodel.collect import news, tickers
from newsmodel.modeling import backtest, dataset
from newsmodel.modeling import train as training


def fetch_news():
    news.main()


def tag_submit():
    tickers.submit()


def tag_collect():
    tickers.collect()


def train():
    dataset.build()
    training.main()
    backtest.main()
