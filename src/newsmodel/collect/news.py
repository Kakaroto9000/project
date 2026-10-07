"""
Job 1: pull news from Event Registry (eventregistry.org) for the outlets in
NEWS_SOURCES and add it to data/raw_news/news_raw.json, where it waits for
Job 2 (articles already in news_tagged.json aren't added again).

Event Registry prices two date ranges differently:
  the last 30 days      -- 1 token per 100 articles   (fetch_recent)
  anything older        -- 5 tokens per 100 articles  (fetch_history, extend_ranges)
Every range is pulled week by week, with its token budget and article limit
spread evenly over the weeks, so every week is covered. Within a week, articles
are sorted by NEWS_SORT_BY. Event Registry has no "tickers" field; Job 2 works
out which companies each article affects.

Progress is written to news_raw_<timestamp>.partial.jsonl as articles
arrive, so a crash keeps what was already pulled.

Run:
    job1-fetch-news                      (or: python -m newsmodel.collect.news)
"""

import json
import logging
import math
import sys
from datetime import date, datetime, timedelta
from functools import cache

from eventregistry import ArticleInfoFlags, EventRegistry, QueryArticlesIter, QueryItems, ReturnInfo

from newsmodel import config

ARTICLE_FIELDS = ReturnInfo(articleInfo=ArticleInfoFlags(
    basicInfo=True,  # publish date and time (dateTime), language, duplicate flag
    bodyLen=-1,      # -1 = the full article body, not a cut-off preview
    body=True, sentiment=True, location=True,
    concepts=False, authors=False, image=False,
))
ARTICLES_PER_REQUEST = 100      # fixed by the eventregistry package
RECENT_DAYS = 30                # Event Registry's cheaper "recent data" window
RECENT_COST_GUESS = 1           # tokens per request until a response reports the real cost
HISTORY_COST_GUESS = 5


class StopOnApiError(logging.Handler):
    """Event Registry reports some problems (too many keywords, plan limits) inside a normal
    response, which the package only logs, so the pull would keep spending tokens on failed
    requests. This turns those log messages into an error that stops the run."""
    def emit(self, record):
        if record.levelno >= logging.ERROR:
            raise RuntimeError(record.getMessage())


# --- What to ask for ---

def resolve_sources(er: EventRegistry) -> list:
    """Event Registry's ID for each outlet in NEWS_SOURCES (free lookups).
    Outlets it doesn't know are printed and skipped."""
    print("Resolving news sources:")
    uris = []
    for name in config.NEWS_SOURCES:
        uri = er.getSourceUri(name)
        print(f"  {name:20s} -> {uri or 'NOT FOUND (skipped)'}")
        if uri:
            uris.append(uri)
    if not uris:
        raise SystemExit("None of NEWS_SOURCES resolved to an Event Registry source.")
    return uris


@cache  # each file is read once per run, not once per week
def load_categories(path) -> tuple:
    """Category IDs from a categories file: one per line, anything after # is a note."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return tuple(uri for line in lines if (uri := line.split("#", 1)[0].strip()))


def categories_for(week_start: date) -> tuple:
    """The categories to filter a week by: iptc categories only exist from
    NEWS_IPTC_START_DATE, so older weeks use the broad news/ ones."""
    if week_start < date.fromisoformat(config.NEWS_IPTC_START_DATE):
        return load_categories(config.NEWS_CATEGORIES_EARLY_FILE)
    return load_categories(config.NEWS_CATEGORIES_FILE)


def build_query(source_uris: list, start: date, end: date) -> QueryArticlesIter:
    """The query for one week: any of our outlets AND any of our categories
    AND any of our keywords, in NEWS_LANG, with copies of the same article left out."""
    return QueryArticlesIter(
        sourceUri=QueryItems.OR(source_uris),
        categoryUri=QueryItems.OR(list(categories_for(start))),
        keywords=QueryItems.OR(config.NEWS_KEYWORDS),
        keywordsLoc="body,title",          # a keyword in the headline or the text counts
        lang=config.NEWS_LANG,
        dateStart=start,
        dateEnd=end,
        isDuplicateFilter="skipDuplicates",  # one outlet republishing another's article
    )


# --- Pulling ---

def week_slices(start: date, end: date) -> list:
    """(start, end) pairs of NEWS_SLICE_DAYS days covering the range. A slice never
    crosses New Year, because a query spanning two years costs double."""
    slices = []
    while start <= end:
        slice_end = min(
            start + timedelta(days=config.NEWS_SLICE_DAYS - 1),
            date(start.year, 12, 31),
            end,
        )
        slices.append((start, slice_end))
        start = slice_end + timedelta(days=1)
    return slices


def pull_week(er, source_uris, week_start, week_end, requests, items, partial_file) -> int:
    """Pull up to `requests` pages (100 articles each) for one week into `items` and the
    progress file. Returns how many articles arrived."""
    got = 0
    articles = build_query(source_uris, week_start, week_end).execQuery(
        er,
        sortBy=config.NEWS_SORT_BY,
        maxItems=requests * ARTICLES_PER_REQUEST,
        returnInfo=ARTICLE_FIELDS,
    )
    for article in articles:  # the package fetches the next page as this loop needs it
        items.append(article)
        got += 1
        partial_file.write(json.dumps(article) + "\n")
        if len(items) % config.NEWS_CHECKPOINT_EVERY == 0:
            partial_file.flush()  # make sure the progress file is really on disk
    return got


def pull_weeks(er, source_uris, start, end, token_budget, max_articles, cost, partial_file):
    """Pull start..end week by week; by each week's end, at most its share of days' worth of
    the token budget and article limit is used, and quiet weeks leave room for later ones.
    Returns (articles, tokens spent)."""
    total_days = (end - start).days + 1
    items, spent = [], 0.0
    for week_start, week_end in week_slices(start, end):
        share = ((week_end - start).days + 1) / total_days  # share of days covered so far
        requests = min(
            int((token_budget * share - spent) / cost),
            int((max_articles * share - len(items)) / ARTICLES_PER_REQUEST),
        )
        if requests < 1:
            continue

        week_tokens = 0.0
        # Event Registry occasionally answers a normal week with nothing (seen for two
        # February 2026 weeks); one retry recovered them, so try once more before moving on.
        for attempt in range(2):
            got = pull_week(er, source_uris, week_start, week_end, requests, items, partial_file)
            reported = float(er.getLastHeader("req-tokens") or 0)  # real cost of the last request
            if reported > 0:
                cost = reported
            week_tokens += max(1, math.ceil(got / ARTICLES_PER_REQUEST)) * cost  # empty still costs 1
            if got:
                break
            if attempt == 0:
                print(f"  {week_start} to {week_end}: 0 articles, retrying once")
        spent += week_tokens
        print(f"  {week_start} to {week_end}: {got:5d} articles, {week_tokens:4.0f} tokens"
              f"  (total {len(items)} articles, {spent:.0f}/{token_budget:.0f} tokens)")
    return items, spent


# --- Date ranges ---

def fetch_recent(er, source_uris, partial_file):
    """The last RECENT_DAYS days, with NEWS_RECENT_TOKEN_BUDGET tokens (for a fresh pull)."""
    today = date.today()
    start = today - timedelta(days=RECENT_DAYS - 1)
    print(f"\nRecent news, {start} to {today}:")
    return pull_weeks(er, source_uris, start, today, config.NEWS_RECENT_TOKEN_BUDGET,
                      config.NEWS_RECENT_MAX_ARTICLES, RECENT_COST_GUESS, partial_file)


def fetch_history(er, source_uris, token_budget, partial_file, end=None, max_articles=None):
    """NEWS_START_DATE up to where the recent window starts, with `token_budget` tokens
    (for a fresh pull)."""
    start = date.fromisoformat(config.NEWS_START_DATE)
    end = end or date.today() - timedelta(days=RECENT_DAYS)
    max_articles = config.NEWS_HISTORY_MAX_ARTICLES if max_articles is None else max_articles
    print(f"\nHistorical news, {start} to {end} (up to {max_articles} articles):")
    return pull_weeks(er, source_uris, start, end, token_budget,
                      max_articles, HISTORY_COST_GUESS, partial_file)


def extend_ranges(er, source_uris, new, partial_file):
    """Pull each NEWS_EXTRA_RANGES range into `new`, in order, each with all tokens still
    left in the account (minus NEWS_TOKEN_RESERVE). Stops when the tokens run out."""
    for start, end in config.NEWS_EXTRA_RANGES:
        usage = er.getUsageInfo()
        tokens_left = usage["availableTokens"] - usage["usedTokens"] - config.NEWS_TOKEN_RESERVE
        if tokens_left < HISTORY_COST_GUESS:
            print(f"\nOut of tokens, skipping {start} to {end} and later ranges")
            return
        print(f"\n{start} to {end} ({tokens_left} tokens left):")
        items, _ = pull_weeks(er, source_uris, date.fromisoformat(start), date.fromisoformat(end),
                              tokens_left, 10**9, HISTORY_COST_GUESS, partial_file)  # 10**9: no article cap
        new.extend(items)


# --- Saving ---

def merge_articles(new: list, previous: list) -> list:
    """`previous` plus `new`, each article once. Articles already in news_tagged.json are
    left out: news_raw.json only holds articles waiting for Job 2."""
    path = config.TAGGED_NEWS_PATH
    tagged = json.loads(path.read_text()) if path.exists() else []
    seen, items = {a["uri"] for a in tagged}, []
    for article in new + previous:
        if article["uri"] not in seen:
            seen.add(article["uri"])
            items.append(article)
    return items


def save_articles(path, items: list):
    """Save to `path`. Written to a temporary file first and then swapped in,
    so a crash mid-write can't corrupt the only copy."""
    temp_path = path.with_suffix(".json.tmp")
    with open(temp_path, "w") as f:
        json.dump(items, f)
    temp_path.replace(path)  # one step: the file is either all old or all new


def print_report(items: list):
    """Articles and median length per outlet (a few hundred characters = summaries only),
    date range, articles per month, and one sample article to check field names."""
    body_lengths = {}
    per_month = {}
    for it in items:
        src = (it.get("source") or {}).get("title") or "?"
        body_lengths.setdefault(src, []).append(len(it.get("body") or ""))
        month = (it.get("dateTime") or "unknown")[:7]  # "2026-03"
        per_month[month] = per_month.get(month, 0) + 1

    print("\nArticles per source (median body length):")
    for src, lengths in sorted(body_lengths.items(), key=lambda kv: -len(kv[1])):  # biggest first
        median = sorted(lengths)[len(lengths) // 2]
        print(f"  {src:35s} {len(lengths):6d}   median {median} chars")

    dates = [it["dateTime"] for it in items if it.get("dateTime")]
    if dates:
        print(f"\nOldest: {min(dates)}\nNewest: {max(dates)}")
    print("Articles per month: " + ", ".join(f"{m} {n}" for m, n in sorted(per_month.items())))

    if items:
        print("\nSample item (check field names against https://eventregistry.org/documentation):")
        print(json.dumps(items[0], indent=2)[:1500])


def main():
    """job1-fetch-news: pull NEWS_EXTRA_RANGES, add the new articles to news_raw.json,
    and print a report. Whatever was pulled is saved even if the pull stops partway."""
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles crash on
                                                                 # e.g. Arabic outlet names
    logging.getLogger("eventregistry").addHandler(StopOnApiError())
    if not config.EVENTREGISTRY_API_KEY:
        raise SystemExit("Set EVENTREGISTRY_API_KEY as an environment variable before running.")

    config.RAW_NEWS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    partial_path = config.RAW_NEWS_DIR / f"news_raw_{timestamp}.partial.jsonl"

    # The package retries failed requests forever by default; stop after 3 so a
    # server outage ends the run instead of hanging it.
    er = EventRegistry(apiKey=config.EVENTREGISTRY_API_KEY, repeatFailedRequestCount=3)
    print(f"Event Registry usage before: {er.getUsageInfo()}")
    source_uris = resolve_sources(er)
    for path in (config.NEWS_CATEGORIES_EARLY_FILE, config.NEWS_CATEGORIES_FILE):
        print(f"Category filter: {len(load_categories(path))} categories from {path.name}")

    path = config.RAW_NEWS_PATH
    previous = json.loads(path.read_text()) if path.exists() else []
    print(f"\nExtending {path.name} ({len(previous)} articles)")

    # For a fresh pull instead, replace extend_ranges with fetch_recent + fetch_history.
    new = []
    try:
        with open(partial_path, "w") as partial_file:
            extend_ranges(er, source_uris, new, partial_file)
    except Exception as e:
        print(f"\nPull stopped: {e}\nSaving what was pulled before the stop.")

    items = merge_articles(new, previous)
    print(f"\nEvent Registry usage after: {er.getUsageInfo()}")
    save_articles(path, items)
    partial_path.unlink()  # everything in it is now in news_raw.json
    print(f"Saved {len(items)} articles to {path} ({len(new)} pulled now, "
          f"{len(new) + len(previous) - len(items)} repeats dropped)")
    print_report(items)


if __name__ == "__main__":
    main()
