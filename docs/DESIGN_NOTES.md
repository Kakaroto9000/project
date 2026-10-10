# Design notes

Why the project is built the way it is, what hasn't been verified yet, and
what's left to do. For setup and how to run it, see `README.md`.

## Decisions (flag before changing any of these)

1. **Labels are up / down / flat, never a percentage.** The model never
   outputs a number. The flat cutoff is per stock (changed 2026-10-09):
   half of that stock's usual 30-minute move, from its 20-day volatility
   (`FLAT_THRESHOLD_SCALED`, `FLAT_THRESHOLD_FRACTION`), so "up" means
   "moved more than usual for this stock" -- 0.2% is a real move for SPY
   but noise for NVDA. A fixed `FLAT_THRESHOLD_PCT` is the alternative.
2. **The model trains on the text, when the news came out, and how prices
   moved before it** (changed 2026-10-09). TF-IDF over headline + article
   body; the stock's and SPY's move over the hour before the news, volume
   in that hour vs. usual, the 1- and 5-day trend and 20-day volatility;
   `news_session`, `role`, outlet, day of week, companies per article and
   this one's rank -- into logistic regression. Nothing from after the
   news is an input. `sentiment_score` (Event Registry's own -1..1 score)
   is stored in the dataset but deliberately not used as a feature.
3. **The label is the 30-minute move from when you could first trade on the
   news** (`HORIZON_MINUTES`, changed 2026-10-09 from the next daily close).
   "When the news came out" is the outlet's publish time (`dateTimePub`,
   `NEWS_TIME_FIELD`), not when Event Registry picked it up (`dateTime`:
   median 6 minutes later, Reuters ~17, 12% over 30). Decided 2026-10-09:
   the user will get articles from a faster feed for live trading, so the
   backtest is only realistic with such a feed, not with Event Registry.
   During market hours it runs from the last price before the news to 30
   minutes later, from 1-minute bars. News in the last 30 minutes of a
   session runs to the 16:00 close instead (a shorter window, since big
   news often drops just before the close), if at least 10 minutes are
   left (`LATE_NEWS_MIN_MINUTES`); the window length is an input
   (`label_minutes`) and the per-stock cutoff shrinks with it. News while
   the market is closed (or after 15:50) runs from the next official
   opening price to 10:00 (`market_time.py` has the rules), so Saturday
   news is Monday 9:30 → 10:00. The jump from the previous close to that
   open isn't in the label -- it couldn't be traded -- but is an input
   (`open_gap_pct`), so the model knows how much is already priced in.
   Weekends and holidays have no prices, so they're skipped. Tradeoff: for
   market-hours news the entry price is up to 1 minute before the news,
   so those trades are slightly optimistic.
   Articles published before 2023 (`MIN_PUBLISH_DATE`) are skipped: they're
   archive pages Event Registry re-crawled years later.
   Every article also gets a row for each market index fund
   (`INDEX_TICKERS`: SPY, QQQ, DIA, IWM, plus SMH chips, XLE energy and USO
   oil; role "index"), even if Job 2
   tagged no company -- added 2026-10-09 so the model learns how news moves
   the whole market and key sectors. That's ~1.35M index rows next to ~330k company rows,
   and articles from the same hour share the same index move.
4. **Train/validation/test split is by date, never shuffled.** Shuffling would
   let the model learn from news published after its test examples.
5. **One article affecting several companies becomes one row per company.**
   Same text, different price outcome per ticker. Keeping only the first
   ticker was a real bug once; don't reintroduce it.
6. **Simple model first.** No neural net until logistic regression beats the
   "always predict the most common label" baseline.
7. **The backtest checks the signal, not portfolio returns.** It sums
   per-trade returns after slippage; there's no position sizing or capital
   limit.
8. **News comes from a fixed list of outlets** (`NEWS_SOURCES`, 10 US-focused
   outlets: Reuters, CNBC, AP, Business Insider, Fortune, Axios, The Verge,
   TechCrunch, FierceBiotech, OilPrice.com), all checked on 2026-10-03 to
   return full text on this plan. Bloomberg and WSJ were removed that day:
   articles carry `userHasPermissions: False` on this plan, so every one is
   cut to ~300 characters, yet they took 57% of the slots (a higher plan
   might unlock them). The recent part (Sep 4 – Oct 3, 9,896 articles) was
   pulled before that change -- it still includes Bloomberg/WSJ and lacks
   the five added outlets -- and is reused rather than re-pulled; history
   is pulled with the new list and ends the day before it.
   Dropped for the first run, all listed in `config.py` for later: general
   news and Asia outlets (they crowd out specific finance stories), Yahoo
   Finance (mostly republished copies that get skipped), MarketWatch and
   Barron's (same owner as WSJ; absent from all 2026-10-03 test pulls) and
   FT (UK focus; mostly letters in the tests). Filtered to the
   categories in `news_categories.txt`, and
   pulled week by week in two parts because Event Registry
   prices them differently: the last 30 days (1 token per 100 articles,
   `NEWS_RECENT_TOKEN_BUDGET` = 100 → ~10,000 articles) and
   `NEWS_START_DATE` to 30 days ago (5 tokens per 100, capped at
   `NEWS_HISTORY_MAX_ARTICLES` = 100,000, assuming the 10k-token plan doubled
   the 5k plan's 50,000 limit). Each part spreads its
   token budget and article limit evenly over its weeks, and whichever is
   tighter decides. Spreading the article limit matters: hitting it
   mid-pull would otherwise leave the last weeks empty without any error.
   The recent weeks end up with more articles per week than the older ones.
   Articles must contain at least one of `NEWS_KEYWORDS` (62 finance and
   technology terms, headline or text; the plan allows 80 words in total,
   counting each word of a phrase), and within each week they're sorted by relevance to
   those keywords (`rel`). Tested on 2026-10-03 (3 tokens): share counts
   are empty for these outlets, and without keywords every sort (`rel`,
   `socialScore`) collapses to time order, so a week's share would come from
   just a few hours at one end of it. With keywords, relevance scores vary
   and a week's articles spread across all days; one test week had ~3,200
   matching articles, so the pull keeps roughly the most relevant 45%.
   Tradeoff: relevance favors articles with many finance words, so long
   columns and multi-company roundups rank high. Weeks never cross New
   Year, which could double a request's cost. Job 1
   reads the real cost of each request and prints account usage before and
   after. High-volume outlets get a bigger share; the report shows counts
   per outlet and per month. Articles Event Registry marks as copies of
   another article are skipped.
9. **Tickers are inferred by Claude Haiku 4.5** (Job 2, Batch API, ~$130 for
   ~195k articles with the prompt cached), because Event Registry doesn't tag articles with tickers.
   Compared on the same 200 random articles (2026-10-06) with a local Gemma 4
   12B ($0): Haiku tagged more articles (130 vs 121), used all 7 roles instead
   of mostly "direct", and got more tickers right (EIX not EC, DASH not DOOD).
   Both gave some non-US or made-up tickers; the prompt's ticker rules target
   those, plus Haiku's own slips (SHOP for Shoppers Stop, XLE, WAYMO). Each ticker gets a role (direct, competitor, ...). Capped at 20
   companies per article to keep one broad story from flooding the dataset.
   Wrong tickers aren't filtered in Job 2; they find no prices.
10. **One article per event.** Before tagging, Job 2 keeps only the earliest
   article for each Event Registry `eventUri`, so several outlets covering
   the same story count once (and only one gets paid for). The earliest is
   kept because that's when the news reached the market. Articles without
   an `eventUri` are all kept. Note: follow-up articles on later days of
   the same event are dropped too.

## Verified with real Event Registry calls (2026-10-03)

- Field names: `title`, `dateTime`, `source` (with `title`), `sentiment`,
  `eventUri`, `url`, `lang`, `isDuplicate`, `relevance` are in real
  responses (`body` and `location` weren't requested in the test).
- A request in the last-30-days window costs 1 token; source and category
  lookups and usage checks are free.
- All `NEWS_SOURCES` entries resolve to the intended outlet.

## Not yet verified

- **Archive cost.** Assumed 5 tokens per request for older weeks, per the
  pricing page; Job 1 corrects this from each response. If an account
  limit is hit, Event Registry refuses the request (401/403) and Job 1
  stops with an error; everything pulled so far stays in the
  `.partial.jsonl` file. Network or server errors are retried 3 times, then
  stop the same way. A re-run starts over (no resume yet).
- **`location` field.** Requested but not yet seen in a real response.
- **Weekly volume without Bloomberg and WSJ.** History aims for ~2,850
  articles a week from Reuters, CNBC and the three niche outlets; if they
  publish fewer matching articles, weeks come in short (fewer articles,
  fewer tokens). The per-week lines show it.

## Open problems (in priority order)

1. **Tagging quality.** Not yet checked on real articles: how many tickers
   per article, how often each role is used, how many tickers are wrong.
2. **Market-relative labels.** Label by the stock's return minus the market's
   (e.g. SPY) over the same window, so a market-wide move isn't credited to
   the news.
3. **Walk-forward validation.** Replace the single date split with several
   rolling train/test windows and average the results.
4. **Ticker concentration.** A few heavily covered companies (e.g. Tesla) may
   dominate. Cap rows per ticker, or report accuracy with and without the
   top tickers.
5. **More features, one at a time**, each compared against the text-only
   baseline: price momentum before the news, number of tickers per article,
   per-source track record.

## Next step

Run Job 2 on a few hundred articles first and read the answers (open
problem 1), then the full run.
