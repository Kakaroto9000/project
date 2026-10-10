"""
Central config. Fill in your API keys as environment variables before running:

    export EVENTREGISTRY_API_KEY="your_key_here"
    export ALPACA_API_KEY="your_key_here"
    export ALPACA_SECRET_KEY="your_key_here"
    export ANTHROPIC_API_KEY="your_key_here"   # Job 2 (Claude ticker tagging)

Everything else here is a tunable knob. Change ONE thing at a time when
experimenting (e.g. just HORIZON_MINUTES) so you know what caused a result
to change.
"""

import os
from pathlib import Path

# --- Paths ---
# Anchored to the repo root (not the current directory), so jobs write to
# the same data/ and models/ no matter where you launch them from.
# Requires an editable install (`pip install -e .`) so this file stays in the repo.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_NEWS_DIR = DATA_DIR / "raw_news"
RAW_PRICES_DIR = DATA_DIR / "raw_prices"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RAW_NEWS_PATH = RAW_NEWS_DIR / "news_raw.json"  # pulled articles waiting for Job 2; Job 1 adds to it
TAGGED_NEWS_PATH = PROCESSED_DIR / "news_tagged.json"  # every tagged article; Job 2 moves articles here

# --- API credentials (read from environment, never hardcode these) ---
EVENTREGISTRY_API_KEY = os.environ.get("EVENTREGISTRY_API_KEY", "")
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # paper trading endpoint

# --- News (Event Registry -- eventregistry.org / newsapi.ai) ---
# Pulled in two parts, priced differently by Event Registry:
#   recent  -- the last 30 days: 1 token per 100 articles, plan limit 100k articles
#   history -- NEWS_START_DATE to 30 days ago: 5 tokens per 100 articles
# Recent goes first with its own budget; history gets the tokens left over.
# Both are pulled week by week with their budget spread evenly over the weeks.
NEWS_START_DATE = "2026-01-01"       # first iteration: 2026 only; widen next month
NEWS_LANG = "eng"
NEWS_TOKEN_BUDGET = 10000            # total tokens to spend (10k-token plan)
NEWS_RECENT_TOKEN_BUDGET = 100       # of which on the last 30 days (~10,000 articles)
NEWS_RECENT_MAX_ARTICLES = 100000
NEWS_HISTORY_MAX_ARTICLES = 100000
NEWS_SLICE_DAYS = 7
NEWS_SORT_BY = "rel"                 # relevance to NEWS_KEYWORDS. Tested 2026-10-03: share counts are empty
                                     # for these outlets, and without keywords every sort collapses to time order
NEWS_CHECKPOINT_EVERY = 1000         # articles between progress saves

# Extending an existing pull: Job 1 pulls these date ranges in order and merges the new
# articles into news_raw.json (dropping repeats by article ID). Each range may use all tokens still left in the account (minus the
# reserve), spread evenly over its weeks, so later ranges get whatever earlier ones
# didn't need.
NEWS_EXTRA_RANGES = [
    ("2026-02-05", "2026-02-11"),    # came back empty on 2026-10-03 (Event Registry glitch)
    ("2026-02-19", "2026-02-25"),    # same
    ("2026-03-12", "2026-03-18"),    # only 900 of ~2,000
    ("2025-01-01", "2025-12-31"),
    ("2024-01-01", "2024-12-31"),    # whatever tokens are left after 2025
]
NEWS_TOKEN_RESERVE = 10              # never spend the last few tokens (a refused request stops the run)

# Outlets to pull from, as Event Registry source IDs (usually the outlet's domain).
# All are pulled together in each weekly query.
NEWS_SOURCES = [
    # business and market news (all checked 2026-10-03: full text on this plan)
    "reuters.com",
    "cnbc.com",
    "apnews.com",
    "businessinsider.com",
    "fortune.com",
    "axios.com",
    # tech
    "theverge.com",        # consumer tech
    "techcrunch.com",      # tech companies, deals, AI
    # niche industries
    "fiercebiotech.com",   # biotech: trials, FDA decisions, deals
    "oilprice.com",        # energy
]
# Not in this run -- candidates for a later run:
#   general news:  "bbc.com", "aljazeera.com", "theguardian.com", "nytimes.com"
#   Asia-focused:  "asia.nikkei.com", "scmp.com"
#   overlapping finance: "barrons.com" (same owner as WSJ, absent in 2026-10-03 test
#                        pulls), "ft.com" (UK focus, mostly letters in tests)
#   "finance.yahoo.com" -- mostly republishes other outlets, so many of its articles
#                          are copies that get skipped
#   other niche: "statnews.com", "endpts.com" (biotech, more paywalled), "rigzone.com" (energy)
#   "bloomberg.com", "wsj.com" -- not licensed on this plan (userHasPermissions = False):
#                                 every article came back as a ~300-character teaser
#   "marketwatch.com", "arstechnica.com" -- open, but only ~1 article per 2 days

# Finance keywords, matched in headline and text. They do two things: an article
# must contain at least one to be returned, and they're what the "rel" sort ranks by
# (more matches / matches in the headline should rank higher).
# Plan limit: 80 WORDS in total -- a phrase like "interest rates" counts as 2.
NEWS_KEYWORDS = [
    # company results
    "earnings", "revenue", "profit", "quarterly results", "guidance", "outlook", "forecast",
    # stock and analyst
    "shares", "stock", "upgrade", "downgrade", "price target", "dividend", "buyback",
    # deals and corporate events
    "merger", "acquisition", "takeover", "IPO", "bankruptcy", "layoffs",
    "lawsuit", "settlement", "antitrust", "recall", "contract",
    # regulation and approvals
    "SEC", "FDA approval", "clinical trial", "regulator", "tariff", "sanctions",
    # macro and markets
    "Federal Reserve", "interest rates", "inflation", "oil prices", "supply chain",
    "OPEC", "shortage", "strike",
    # technology
    "artificial intelligence", "AI", "semiconductor", "chip", "data center", "cloud", "software",
    "cybersecurity", "data breach", "patent", "battery", "electric vehicle", "self-driving",
    "robotics", "quantum computing", "satellite", "gene therapy", "biotech", "renewable energy",
    "nuclear power", "refining", "breakthrough", "product launch",
]

# Event Registry categories to keep, one per line (an article is kept if it's in any of them).
NEWS_CATEGORIES_FILE = Path(__file__).parent / "news_categories.txt"
# Event Registry only started assigning the iptc/ categories above around this date
# (tested 2026-10-03: a January iptc-filtered query returned nothing). Weeks starting
# before it use the broader news/ categories in NEWS_CATEGORIES_EARLY_FILE instead.
NEWS_IPTC_START_DATE = "2026-05-29"
NEWS_CATEGORIES_EARLY_FILE = Path(__file__).parent / "news_categories_early.txt"

# Optional allow-list applied AFTER Job 2 infers tickers (see
# modeling/dataset.py) -- empty = keep every ticker Job 2 tagged. Populate
# this (e.g. ["AAPL", "TSLA", "GM"]) to narrow the final dataset to tickers
# you care about, without changing the news query itself.
TICKERS_UNIVERSE = []

# --- What the model predicts, and what it sees first (market_time.py, modeling/dataset.py) ---
# The label is the price move over HORIZON_MINUTES from the first moment you could trade
# on the news: right away during market hours, otherwise from the next official opening
# price (the jump to that open is an input, not part of the label). Before predicting,
# the model sees how the stock and the market moved over the PRE_NEWS_MINUTES before the news.
# News in the last HORIZON_MINUTES of a session is measured to the 16:00 close instead
# (a shorter window -- big news often drops just before the close), as long as at least
# LATE_NEWS_MIN_MINUTES are left; later news waits for the next open.
#
# Labels are a 3-way category: up / down / flat. The model predicts ONE OF
# THESE THREE LABELS -- it never predicts a percentage/magnitude. The
# threshold below only decides where "flat" starts; it's a labeling rule,
# not something the model outputs.
# Which Event Registry time counts as "when the news came out": "dateTimePub" (when the outlet
# published it) or "dateTime" (when Event Registry picked it up -- median 6 min later, Reuters
# ~17). Publish time measures the real reaction, but the backtest is only realistic with a news
# feed that delivers articles that fast (Event Registry does not).
NEWS_TIME_FIELD = "dateTimePub"
# Articles published before this are skipped: archive pages Event Registry re-crawled years
# later (a 2015 Verge story picked up in 2026). Everything published from here on is kept,
# however late it was picked up.
MIN_PUBLISH_DATE = "2023-01-01"

# Market and sector funds every article is also checked against, whatever Job 2 tagged: one
# extra row per article per fund, with role "index" (unless Job 2 tagged the fund itself).
# Market: SPY = S&P 500, QQQ = Nasdaq-100, DIA = Dow Jones, IWM = Russell 2000 (small companies).
# Sectors: SMH = chip makers, XLE = energy companies (S&P 500 energy stocks),
# USO = crude oil itself (follows oil futures, not companies). [] turns it off.
INDEX_TICKERS = ["SPY", "QQQ", "DIA", "IWM", "SMH", "XLE", "USO"]
HORIZON_MINUTES = 30
LATE_NEWS_MIN_MINUTES = 10          # news up to 15:50 is measured to the close
PRE_NEWS_MINUTES = 60
INTRADAY_BAR_MINUTES = 1             # price resolution: a price "at" a time is the last bar
                                     # finished by then, so up to this many minutes old
MARKET_TICKER = "SPY"                # the market's move over the same window, and the trading calendar
TREND_DAYS = (1, 5)                  # stock's return over this many trading days before the news
VOLATILITY_DAYS = 20                 # stock's typical daily move, over this many trading days

# A move smaller than the cutoff (in either direction) gets labeled "flat"
# instead of forced into "up" or "down". Without this, tiny random noise
# around 0% gets labeled as a real move, which teaches the model to chase
# noise.
#
# FLAT_THRESHOLD_SCALED = True: each stock gets its own cutoff, so "up" means "moved
# more than usual for this stock" -- 0.2% is a real move for SPY but noise for NVDA.
# The cutoff is FLAT_THRESHOLD_FRACTION of the stock's usual move over the label's window
# (HORIZON_MINUTES, or less for news just before the close): its VOLATILITY_DAYS daily
# volatility scaled down to that window (daily x sqrt(minutes / 390), the usual
# random-walk scaling). With 0.5 that's roughly
# SPY ~0.15%, NVDA ~0.4%, a small biotech 1%+.
# FLAT_THRESHOLD_SCALED = False: one fixed FLAT_THRESHOLD_PCT for every stock.
# Pick between settings by the VALIDATION results, never the test results.
FLAT_THRESHOLD_SCALED = True
FLAT_THRESHOLD_FRACTION = 0.5
FLAT_THRESHOLD_PCT = 0.2             # fixed mode, and the fallback for a stock with under a week of history

# --- AI ticker tagging (collect/tickers.py, Job 2) ---
# Each article's headline + a slice of its body goes to Claude (Anthropic Batch API, half
# the price of normal calls), which returns the tickers the news affects, each with a role.
# Chosen 2026-10-06 over a local Gemma 4 12B: in a 200-article test Claude got more
# tickers right and used the roles properly (comparison in docs/DESIGN_NOTES.md).
TAGGING_MODEL = "claude-haiku-4-5"
ARTICLE_CHARS_FOR_TAGGING = None     # how much of the body to send; None = the whole article
                                     # (avg ~840 tokens; the prompt is 1,329 of the ~2,200 per request)
# Anthropic caps one batch at 256 MB, but a ~180 MB upload (20,000 whole articles) kept
# failing over a home connection (2026-10-06), so batches are kept small: 5,000 articles
# is ~40 MB. Each batch writes the cached prompt once, which costs almost nothing.
TAGGING_BATCH_SIZE = 5000

# Ceiling on how many companies the model may return per article -- not a target:
# it lists only the ones it judges important, often 0-5. Without a ceiling, a
# broad-impact article (a Fed rate decision, a market-wide selloff) could list
# 100+ companies, raising cost and risking an answer cut off mid-JSON,
# which would drop ALL of that article's tickers.
MAX_COMPANIES_PER_ARTICLE = 20

# --- Local ticker tagger (modeling/tagger.py) ---
# A local model that learns to copy Claude's tags, so new articles can be tagged instantly
# and for free. Companies Claude tagged in fewer training articles than TAGGER_MIN_ARTICLES
# aren't learned (100 -> ~420 companies, 79% of Claude's tags, checked 2026-10-09).
TAGGER_MIN_ARTICLES = 100
TAGGER_MAX_FEATURES = 50000          # words/phrases it can use (names need a big vocabulary)
TAGGER_C = 1.0                       # how closely each yes/no model may fit its examples
TAGGER_JOBS = 8                      # CPU cores used for training

# --- Repeats of the same story (modeling/dataset.py) ---
# Several outlets often report the same news within hours (one Event Registry event). Job 2
# tags all of them; for training, True keeps only the first report and drops other reports
# of that event published within STORY_REPEAT_HOURS of it -- otherwise one Fed decision
# covered by 5 outlets is 5 rows with the same label, and later copies enter after the
# market already reacted. Later follow-ups are kept (big events can merge separate stories
# over several days, checked 2026-10-04). Set False to train on every report.
DATASET_SKIP_STORY_REPEATS = True
STORY_REPEAT_HOURS = 24

# --- Text feature (headline + article body) ---
# TF-IDF vocabulary size. Larger = more words the model can weigh, but also
# more room to overfit a short, likely-one-regime dataset. Start here.
TEXT_MAX_FEATURES = 5000

# Category values (an outlet, a role, ...) seen in fewer training rows than this are
# grouped together as "infrequent" instead of getting their own weight.
MIN_CATEGORY_ROWS = 20

# --- Train/val/test split (chronological, never shuffled) ---
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
# test fraction is whatever remains (0.15)

# --- Backtest frictions (be pessimistic here, not optimistic) ---
COMMISSION_PER_TRADE_PCT = 0.0   # Alpaca is commission-free on stocks, but keep
                                  # the knob in case you add other brokers/assets
SLIPPAGE_PCT = 0.05               # assume you don't get the exact printed price
