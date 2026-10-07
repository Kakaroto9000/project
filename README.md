# News â†’ Price Direction Model

Pull historical news from Event Registry, have Claude figure out which
companies/tickers each article affects, line that up with historical
prices, train a model to predict whether the stock goes up / down / flat
afterwards, and backtest it â€” before ever touching real or even paper money.

## Setup

```bash
pip install -e .        # installs dependencies + the three job commands
```

The `-e` (editable) matters: data paths are anchored to this repo
(`config.PROJECT_ROOT`), so jobs read/write `data/` and `models/` here no
matter which directory you launch them from.

API keys, as environment variables (see `src/newsmodel/config.py`):

| Variable | Used by | Cost |
| --- | --- | --- |
| `EVENTREGISTRY_API_KEY` | Job 1 (news) | your Event Registry plan |
| `ANTHROPIC_API_KEY` | Job 2 (ticker tagging) | ~$130 for ~195k articles (Batch API, prompt cached) |
| `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` | prices.py | free tier is enough |

## Layout

```text
pyproject.toml              dependencies + job commands
docs/DESIGN_NOTES.md        decisions, unverified items, open problems
src/newsmodel/
  config.py                 every tunable knob + data paths
  jobs.py                   the 3 jobs, one function each
  collect/                  talks to external APIs
    news.py                 Event Registry  -> raw news
    tickers.py              Claude batch    -> which tickers each article affects
    prices.py               Alpaca          -> daily price bars
  modeling/                 local only, no network
    dataset.py              news + prices   -> labeled up/down/flat rows
    train.py                text + news timing -> logistic regression
    backtest.py             test-split trades after slippage
data/raw_news/              news_raw.json -- pulled articles waiting to be tagged (Job 1)
data/raw_prices/            prices_*.json (prices.py)
data/processed/             news_tagged.json (Job 2),
                            labeled_dataset.parquet, backtest_trades.csv (Job 3)
models/                     baseline_model.joblib (Job 3)
```

## Run order â€” 3 separate jobs

```bash
job1-fetch-news     # Event Registry -> data/raw_news/news_raw.json
job2-tag-submit     # sends the waiting articles to Claude as batches and exits
job2-tag-collect    # RE-RUN until the batches are done -> news_tagged.json
python -m newsmodel.collect.prices     # Alpaca prices for the tagged tickers
job3-train          # dataset -> model -> backtest; local, safe to re-run anytime
```

Job 2 is two commands because Claude's Batch API works in the background
(usually done within an hour, max 24h): job2-tag-submit sends the articles,
and job2-tag-collect checks on them; once they're finished it saves the
tagged news. Prices are fetched separately afterwards; tickers Claude got
wrong simply find no prices.

Each stage also runs on its own, e.g. after changing only
`TEXT_MAX_FEATURES`:

```bash
python -m newsmodel.modeling.train
```

## Changing what counts as "affected"

The instruction Claude gets for every article is `TAGGING_SYSTEM_PROMPT` in
`src/newsmodel/collect/tickers.py`. Edit it to make tagging stricter or
broader. `MAX_COMPANIES_PER_ARTICLE` (config.py, default 20) caps how many
companies one article can be tagged with â€” it bounds cost and keeps a
single broad article from dominating the dataset. Try prompt changes on a
few hundred articles before a full run.

## What to check at each step

- **After Job 1**: what each entry in `NEWS_SOURCES` resolved to (a wrong
  match means fix the ID), the tokens used per week and account usage before
  and after, articles and median body length per source (a few hundred
  characters means that outlet only gives snippets), articles per month
  (roughly even, with more in the last 30 days; an empty stretch means a
  plan limit was hit early), and the printed sample item â€” confirm the field
  names match what `collect/news.py` expects.
- **After Job 2**: "X/Y articles tagged with at least one ticker". Very
  low means the source/category mix or the prompt needs adjusting (`NEWS_KEYWORDS` can narrow
  noisy sources).
- **After Job 3, dataset step**: the up/down/flat distribution. "flat" near
  0% â†’ `FLAT_THRESHOLD_PCT` too tight; near 90%+ â†’ too wide. Also note the
  majority-class share â€” that's the baseline the model must beat. Also check
  "When the news came out": the split between weekend, pre-market, market
  hours and after hours.
- **After Job 3, train step**: **test accuracy vs. the naive "always
  predict the majority class" baseline**. Not clearly above it is a real
  finding. Next things to try, in order of effort: different
  `HORIZON_DAYS`, different `FLAT_THRESHOLD_PCT`, per-ticker/sector
  breakdown, different `TEXT_MAX_FEATURES` or adding `sentiment_score` back
  as a feature â€” and only then a more complex model.
- **After Job 3, backtest step**: open `backtest_trades.csv` and look for a
  few huge wins/losses dominating the result. In the per-timing breakdown,
  if `market_hours` trades carry most of the profit, be skeptical â€” those
  entries use the previous close, which you couldn't actually trade at.

## Not built yet (by design)

- No real or paper order execution.
- No position sizing, capital constraints, or simultaneous-position handling.
- No live news-polling loop.

Once backtest results hold up across multiple time periods, the next step
is a live loop: pull new articles, tag and score them with this same
pipeline, and submit paper trades through Alpaca.
