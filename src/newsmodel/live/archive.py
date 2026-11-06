"""
Every article the live app collects with a body over LIVE_ARCHIVE_MIN_CHARS,
kept for training later -- including the ones it skips (stocks outside the
watchlist) and ones it saw too late to trade. Headline-only flashes and short
press releases are left out. The model's training data stops in October 2026;
this keeps the history growing for free from then on.

Saved as data/raw_news/live/<YYYY-MM-DD>.jsonl (the UTC day the article reached
us), one article per line, in the same shape as Event Registry's articles, so
the existing pipeline reads them unchanged:
    uri, title, body, url, source.title, dateTimePub (published), dateTime
    (when it reached us), lang, plus symbols (Alpaca's own tickers, if any) and
    collected_by = "live"
Append-only and separate from data/live/live.db, so clearing the app's data
never loses articles.

To train on them, put them in Job 2's queue and run the usual steps:
    python -m newsmodel.live.archive               # archive -> data/raw_news/news_raw.json
    job2-tag-submit / job2-tag-collect              # tickers (Claude), as for Event Registry news
    python -m newsmodel.collect.prices, collect.universe, modeling.dataset, ...
Articles already tagged or already queued are never added twice.

    python -m newsmodel.live.archive --backfill    # first also copy what's in live.db
    python -m newsmodel.live.archive --prune       # drop archived articles under the length limit
"""

import argparse
import json
from collections import defaultdict

import pandas as pd

from newsmodel import config

ARCHIVE_DIR = config.RAW_NEWS_DIR / "live"


def zulu(ts) -> str:
    """Event Registry's time format: 2026-11-04T16:17:22Z."""
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def to_item(a: dict, received) -> dict:
    """A live article (live/news.py's fields) in Event Registry's shape."""
    return {
        "uri": a["uri"], "title": a["title"], "body": a["body"] or "", "url": a["url"] or "",
        "source": {"title": a["source"]}, "dateTimePub": zulu(a["published"]), "dateTime": zulu(received),
        "lang": "eng", "isDuplicate": False, "symbols": list(a.get("symbols") or []), "collected_by": "live",
    }


def long_enough(a: dict) -> bool:
    return len(a.get("body") or "") > config.LIVE_ARCHIVE_MIN_CHARS


def save(articles: list, received=None):
    """Append newly collected articles with a long enough body (the caller drops repeats)."""
    articles = [a for a in articles if long_enough(a)]
    if not articles:
        return
    received = pd.Timestamp(received) if received is not None else pd.Timestamp.now(tz="UTC")
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARCHIVE_DIR / f"{received.tz_convert('UTC'):%Y-%m-%d}.jsonl", "a", encoding="utf-8") as f:
        for a in articles:
            f.write(json.dumps(to_item(a, received), ensure_ascii=False) + "\n")


def prune():
    """Rewrite the archive without articles at or under LIVE_ARCHIVE_MIN_CHARS (e.g. after raising it)."""
    kept = dropped = 0
    for path in sorted(ARCHIVE_DIR.glob("*.jsonl")):
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        keep = [l for l in lines if long_enough(json.loads(l))]
        kept, dropped = kept + len(keep), dropped + len(lines) - len(keep)
        if len(keep) < len(lines):
            tmp = path.with_suffix(".tmp")
            tmp.write_text("".join(l + "\n" for l in keep), encoding="utf-8")
            tmp.replace(path)
    print(f"Pruned the archive: kept {kept}, dropped {dropped} with a body of {config.LIVE_ARCHIVE_MIN_CHARS} characters or less")


def load() -> list:
    """Every archived article, each once (the first time it was seen)."""
    items, seen = [], set()
    for path in sorted(ARCHIVE_DIR.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if item["uri"] not in seen:
                seen.add(item["uri"])
                items.append(item)
    return items


def backfill():
    """Copy articles in data/live/live.db that aren't archived yet (e.g. from before the archive existed)."""
    from newsmodel.live.store import Store
    archived = {i["uri"] for i in load()}
    rows = Store().rows("SELECT uri, title, body, source, url, symbols, published, received FROM articles")
    by_day = defaultdict(list)
    for r in rows:
        if r["uri"] not in archived:
            r["symbols"] = json.loads(r["symbols"] or "[]")
            by_day[pd.Timestamp(r["received"]).strftime("%Y-%m-%d")].append(r)
    for day_rows in by_day.values():
        for r in day_rows:  # keep each article's own received time
            save([r], r["received"])
    n = sum(len(v) for v in by_day.values())
    print(f"Backfilled {n} articles from live.db ({len(rows) - n} were already archived)")


def to_queue():
    """Add archived articles to news_raw.json, where Job 2 picks them up."""
    from newsmodel.collect.news import merge_articles, save_articles
    items = load()
    path = config.RAW_NEWS_PATH
    previous = json.loads(path.read_text()) if path.exists() else []
    merged = merge_articles(items, previous)
    config.RAW_NEWS_DIR.mkdir(parents=True, exist_ok=True)
    save_articles(path, merged)
    added = len(merged) - len(previous)
    print(f"{len(items)} archived articles; {added} added to {path.name} for Job 2 "
          f"({len(items) - added} were already tagged or queued). Next: job2-tag-submit")


def summary():
    items = load()
    if not items:
        print("The archive is empty.")
        return
    df = pd.DataFrame({"source": [i["source"]["title"] for i in items],
                       "published": pd.to_datetime([i["dateTimePub"] for i in items], utc=True),
                       "chars": [len(i["body"]) for i in items]})
    print(f"{len(df)} articles, published {df['published'].min():%Y-%m-%d %H:%M} to {df['published'].max():%Y-%m-%d %H:%M} UTC")
    print(df.groupby("source").agg(articles=("chars", "size"), median_chars=("chars", "median")).sort_values("articles", ascending=False).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true", help="first copy articles from data/live/live.db into the archive")
    ap.add_argument("--summary", action="store_true", help="only show what's archived")
    ap.add_argument("--prune", action="store_true", help="drop archived articles at or under LIVE_ARCHIVE_MIN_CHARS")
    args = ap.parse_args()
    if args.prune:
        prune()
    if args.backfill:
        backfill()
    summary()
    if not args.summary:
        to_queue()


if __name__ == "__main__":
    main()
