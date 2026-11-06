# News → Price Direction Model

Pull financial news, line it up with stock prices, and train a model that reads
an article and predicts whether each stock goes up / down / flat over the next
30 minutes. Then backtest it, and watch it live in a desktop app that only ever
**pretends** to trade.

Three parts, each usable on its own:

1. **Data** — historical news (Event Registry), which companies each article
   affects (Claude), and 1-minute prices (Alpaca).
2. **Models** — from a TF-IDF baseline up to one fine-tuned ModernBERT that
   reads an article once and predicts ~1,000 stocks, with honest evaluation
   (chronological splits, costs, leak checks).
3. **Live app** — `NewsTrader.exe`: free news as it comes out, the model's
   calls, what actually happened, and a pretend $100k account.

**Results so far:** no model has beaten trading costs on the test period
(details in `docs/DESIGN_NOTES.md` and the reports in `data/processed/reports/`).
The live app exists to watch that question on fresh data, not to make money.

## Setup

```bash
pip install -e .        # dependencies + the job commands
```

The `-e` (editable) matters: data paths are anchored to this repo
(`config.PROJECT_ROOT`), so everything reads/writes `data/` and `models/` here
no matter which directory you launch it from. The language models need a CUDA
GPU (built on an RTX 4070).

API keys, as environment variables (see `src/newsmodel/config.py`):

| Variable | Used by | Cost |
| --- | --- | --- |
| `EVENTREGISTRY_API_KEY` | Job 1 (historical news) | your Event Registry plan |
| `ANTHROPIC_API_KEY` | Job 2 (ticker tagging) | ~$130 for ~195k articles (Batch API, prompt cached) |
| `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` | prices, live news, live prices | free plan is enough |

## Layout

```text
pyproject.toml              dependencies + commands
README.md                   this file
docs/DESIGN_NOTES.md        decisions, unverified items, open problems
NewsTrader.exe              the live app (built from ui/, not committed)

src/newsmodel/
  config.py                 every tunable knob + data paths
  jobs.py                   Jobs 1-3, one function each
  market_time.py            market hours: when news came out, when its trading window starts/ends
  keep_awake.py             keeps Windows awake during long jobs
  collect/                  talks to external APIs
    news.py                 Event Registry  -> raw news (Job 1)
    tickers.py              Claude batch    -> which tickers each article affects (Job 2)
    prices.py               Alpaca          -> daily + 1-minute bars for tagged tickers
    universe.py             Alpaca          -> the 1,000 most traded stocks + their 1-minute bars
  modeling/                 local only, no network
    dataset.py              news + prices   -> labeled up/down/flat rows (Job 3)
    train.py, backtest.py   TF-IDF + logistic regression baseline and its backtest (Job 3)
    tagger.py               local model that copies Claude's tags (free, instant)
    embed.py, finetune.py   sentence embeddings; per-company fine-tuned ModernBERT
    universe_dataset.py     (article, company) pairs over the 1,000-stock universe
    grid_dataset.py         article x stock label grid for the one-model-for-all-stocks
    news_model.py           that model: ModernBERT reader + per-stock profile and head
    news_baseline.py        same grid with NO text (LightGBM) -- what the text must beat
    news_eval.py            distinct bets, costs, by-day confidence intervals, leak checks
  live/                     the live pretend-trading engine
    news.py                 free news: Alpaca (Benzinga) + RSS of CNBC, Fortune, BI, ...
    market.py               15-minute-delayed prices -> the model's inputs, exactly as in training
    check_market.py         proves market.py reproduces the training grid
    predictor.py            loads a saved news_model*.pt for the watchlist
    engine.py               collect -> wait for prices -> predict -> pretend trade
    store.py                everything in one SQLite file (data/live/live.db)
    server.py               the engine + a local API for the window (127.0.0.1:8765)

ui/                         the window: Tauri (Rust) + React + Tailwind + lightweight-charts
  src/                      pages (Now, History, Portfolio, Log) and components
  src-tauri/                native shell; starts/stops server.py with the window

data/raw_news/              news_raw.json -- pulled articles waiting to be tagged (Job 1)
data/raw_prices/            daily/intraday bars, universe.json, universe_minute*/ (one file per day)
data/processed/             tagged news, datasets, grids (grid/, grid_ext*/), results, reports/
data/live/                  live.db and the engine's logs
models/                     saved models (baseline_model.joblib, tagger.joblib, news_model*.pt)
```

Everything under `data/`, `models/` and the built `.exe` is generated and not committed.

## Part 1 — Data: 3 jobs

```bash
job1-fetch-news     # Event Registry -> data/raw_news/news_raw.json
job2-tag-submit     # sends the waiting articles to Claude as batches and exits
job2-tag-collect    # RE-RUN until the batches are done -> news_tagged.json
python -m newsmodel.collect.prices     # Alpaca prices for the tagged tickers
job3-train          # dataset -> baseline model -> backtest; local, safe to re-run anytime
```

Job 2 is two commands because Claude's Batch API works in the background
(usually done within an hour, max 24h): job2-tag-submit sends the articles,
and job2-tag-collect checks on them; once they're finished it saves the
tagged news. Prices are fetched separately afterwards; tickers Claude got
wrong simply find no prices.

Each stage also runs on its own, e.g. after changing only `TEXT_MAX_FEATURES`:

```bash
python -m newsmodel.modeling.train
```

### Changing what counts as "affected"

The instruction Claude gets for every article is `TAGGING_SYSTEM_PROMPT` in
`src/newsmodel/collect/tickers.py`. Edit it to make tagging stricter or
broader. `MAX_COMPANIES_PER_ARTICLE` (config.py, default 20) caps how many
companies one article can be tagged with — it bounds cost and keeps a
single broad article from dominating the dataset. Try prompt changes on a
few hundred articles before a full run.

## Part 2 — Models over the 1,000-stock universe

```bash
python -m newsmodel.collect.universe [--extended]       # pick the universe, download its 1-minute bars
python -m newsmodel.modeling.universe_dataset           # which companies each article names
python -m newsmodel.modeling.grid_dataset [--extended]  # article x stock labels and inputs
python -m newsmodel.modeling.news_model [--extended] [--named-only | --clean-only | --top 200] [--horizon 120]
python -m newsmodel.modeling.news_eval --extended --name news_model_ext
python -m newsmodel.modeling.news_baseline --extended   # the no-text comparison
```

`news_model.py` trains on the GPU (~30 min per epoch on an RTX 4070), keeps the
epoch with the lowest validation loss, and saves `models/news_model<options>.pt`
plus validation/test predictions next to the grid. `--top 200` learns only the
market/sector funds plus the 200 most traded companies — the live app's watchlist.

Splits are chronological (70 / 15 / 15, never shuffled). Choose settings by
**validation** results; look at test once.

## Part 3 — The live app (pretend trading)

Double-click **`NewsTrader.exe`**. The window starts the Python engine
(`python -m newsmodel.live.server`) and stops it when closed.

What happens to each new article:

1. It arrives from a free source (`LIVE_NEWS_SOURCES`: Alpaca's news API plus
   the RSS feeds of CNBC, Fortune, Business Insider, TechCrunch, The Verge,
   OilPrice), checked every minute.
2. It waits until its 30-minute trading window is over **and** Alpaca's free
   plan allows the prices (all-exchange data is 15 minutes delayed), ~45 min
   after publishing.
3. The model (`LIVE_MODEL_PATH`) gets the same inputs it trained on — only
   prices from before the news — and calls down / flat / up for the 7 funds
   plus the top 200 companies.
4. Every "up" call at least `LIVE_MIN_UP_PROB` likely becomes a **pretend**
   $1,000 buy at the window's entry price, sold at its exit price, with
   `SLIPPAGE_PCT` each way. Articles that reached us too late to have been
   traded are predicted but not traded. No orders are ever sent anywhere.

The pages: **Now** (incoming news, the latest prediction, the one before it vs.
what happened), **History** (every article and the overall scorecard),
**Portfolio** (the pretend account and every trade), **Log**.

Running it without the window: `python -m newsmodel.live.engine` (engine only)
or `python -m newsmodel.live.server` (engine + API). Check the live inputs
still match training after changing anything in `live/market.py`:
`python -m newsmodel.live.check_market`.

### Rebuilding the window

Needs Node.js, Rust and the Visual Studio C++ Build Tools.

```bash
cd ui
npm install
npx tauri build --no-bundle                      # -> src-tauri/target/release/news-trader.exe
cp src-tauri/target/release/news-trader.exe ../NewsTrader.exe
```

For working on the UI: `python -m newsmodel.live.server --no-engine` (serves
the saved data) and `npm run dev` (`http://localhost:1420`, live reload).
The Python path the window uses is at the top of `ui/src-tauri/src/main.rs`
(or set `NEWSTRADER_PYTHONW` / `NEWSTRADER_ROOT`).

## What to check at each step

- **After Job 1**: what each entry in `NEWS_SOURCES` resolved to (a wrong
  match means fix the ID), the tokens used per week and account usage before
  and after, articles and median body length per source (a few hundred
  characters means that outlet only gives snippets), articles per month
  (roughly even, with more in the last 30 days; an empty stretch means a
  plan limit was hit early), and the printed sample item — confirm the field
  names match what `collect/news.py` expects.
- **After Job 2**: "X/Y articles tagged with at least one ticker". Very
  low means the source/category mix or the prompt needs adjusting (`NEWS_KEYWORDS` can narrow
  noisy sources).
- **After the dataset step**: the up/down/flat distribution. "flat" near
  0% → `FLAT_THRESHOLD_FRACTION` too low; near 90%+ → too high. Also note the
  majority-class share — that's the baseline the model must beat. Also check
  "When the news came out": the split between weekend, pre-market, market
  hours and after hours.
- **After training**: **validation and test results vs. the no-text
  baseline and vs. trading costs** — per distinct bet, with by-day confidence
  intervals (`news_eval.py`). Not clearly above them is a real finding.
- **After a backtest**: look for a few huge wins/losses dominating the result,
  and for leaks — any input that could only be known after the trade starts
  (two have been found and fixed so far; see `docs/DESIGN_NOTES.md`).
