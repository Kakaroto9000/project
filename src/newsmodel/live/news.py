"""
New articles for the live app, from free sources only (config.LIVE_NEWS_SOURCES):

  Alpaca            Alpaca's news API: Benzinga stories (full text) and short fiscal.ai
                    press releases, with tickers. Real time, but not a training outlet.
  CNBC, Fortune,    RSS feeds of the outlets the model was trained on (via Event Registry,
  Business Insider, which we can't use any more). Fortune and Business Insider put the
  TechCrunch,       whole article in the feed. CNBC, TechCrunch, The Verge and OilPrice
  The Verge,        only give a summary, so the article page is read once for its text
  OilPrice          (trafilatura) and its publish time. CNBC's feed time is the LAST
                    UPDATE, not the publish time, so for CNBC the page's time is the one
                    that counts.

Left out (2026-11-04): AP (robots.txt disallows its .rss feeds, and they return 403),
Axios (its feed host's robots.txt disallows everything), Reuters (no public feed any more;
the site blocks bots), Google News (robots.txt disallows its RSS; links are redirects,
headlines only), Yahoo Finance (feeds gone, 404), FierceBiotech (site behind a bot
challenge; the FeedBurner copy has one-line summaries and no time zone). Finnhub,
Marketaux, NewsAPI etc. need a key we don't have.

Polite by design: robots.txt is checked for every feed and page, each feed is fetched at
most once per FEED_MIN_SECONDS, and each article page once (remembered by URL).
A source that fails is skipped with a warning; fetch_new never raises because of one.

Text is cleaned to match the training articles: plain text, paragraphs separated by a
blank line. RSS articles have no tickers (symbols = []).

Try it:
    python -m newsmodel.live.news
"""

import calendar
import html
import re
import time
import urllib.robotparser
import warnings
from urllib.parse import urlsplit

import feedparser
import pandas as pd
import requests
import trafilatura
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning, XMLParsedAsHTMLWarning

from newsmodel import config

# outlet -> (its feeds, where the text comes from: "feed" = the feed itself, "page" = the article page)
FEEDS = {
    "CNBC": ([f"https://www.cnbc.com/id/{i}/device/rss/rss.html" for i in (
        100003114,  # top news
        10001147,   # business
        15839069,   # markets
        10000664,   # finance
        15839135,   # earnings
        20910258,   # economy
        19854910,   # technology
    )], "page"),
    "Fortune": (["https://fortune.com/feed/"], "feed"),
    "Business Insider": (["https://feeds.businessinsider.com/custom/all"], "feed"),
    "TechCrunch": (["https://techcrunch.com/feed/"], "page"),
    "The Verge": (["https://www.theverge.com/rss/index.xml"], "page"),
    "OilPrice": (["https://oilprice.com/rss/main"], "page"),
}
USER_AGENT = "Mozilla/5.0 (compatible; newsmodel-paper/0.1; personal paper-trading app)"
ROBOTS_AGENT = "newsmodel-paper"   # the name robots.txt rules are matched against
FEED_MIN_SECONDS = 60              # never fetch the same feed more often than this
PAGE_DELAY_SECONDS = 0.5           # pause between article pages
ROBOTS_MAX_AGE = 24 * 3600         # re-read each site's robots.txt daily
ALPACA_MAX_ARTICLES = 500          # per call (a first call can look back hours)
TIMEOUT = 20
# feed footers the training articles didn't have
FOOTER = re.compile(r"\s*This story was originally featured on Fortune\.com\s*$")

_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT
_feeds = {}    # feed url -> (fetched at, entries)
_pages = {}    # article url -> (publish time or None, text)
_robots = {}   # site -> (fetched at, parser)
_alpaca = None
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


def _key(outlet: str) -> str:
    return re.sub(r"\W+", "", outlet.lower())  # "Business Insider" -> "businessinsider"


def plain(text: str) -> str:
    """HTML or plain text -> plain text, one blank line between paragraphs (as Event Registry)."""
    if not text:
        return ""
    if "<" in text:
        text = re.sub(r"(?i)<br\s*/?>|</(p|div|li|h\d|tr|blockquote|figure|ul|ol|table)>", "\n", text)
        text = BeautifulSoup(text, "lxml").get_text()   # also unescapes &amp; etc.
    else:
        text = html.unescape(text)
    lines = (re.sub(r"\s+", " ", line).strip() for line in text.replace("\xa0", " ").split("\n"))
    return "\n\n".join(line for line in lines if line)


def one_line(text: str) -> str:
    return " ".join(plain(text).split())


def _utc(value) -> pd.Timestamp:
    t = pd.Timestamp(value)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


# --- Politeness ---

def allowed(url: str) -> bool:
    """Whether the site's robots.txt lets us fetch this URL (rules as Python's robotparser:
    no robots.txt = allowed, 401/403 = nothing allowed)."""
    site = "{0.scheme}://{0.netloc}".format(urlsplit(url))
    fetched, parser = _robots.get(site, (0, None))
    if time.time() - fetched > ROBOTS_MAX_AGE:
        parser = urllib.robotparser.RobotFileParser()
        try:
            r = _session.get(site + "/robots.txt", timeout=TIMEOUT)
            if r.status_code in (401, 403):
                parser.disallow_all = True
            elif r.status_code >= 500:
                return False   # try again next time
            elif r.ok:
                parser.parse(r.text.splitlines())
            else:
                parser.allow_all = True
        except requests.RequestException:
            return False
        _robots[site] = (time.time(), parser)
    return parser.can_fetch(ROBOTS_AGENT, url)


def _get(url: str) -> requests.Response:
    if not allowed(url):
        raise PermissionError("robots.txt disallows it")
    r = _session.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r


# --- RSS outlets ---

def _entries(url: str) -> list:
    """A feed's items, fetched at most once per FEED_MIN_SECONDS."""
    fetched, entries = _feeds.get(url, (0, []))
    if time.time() - fetched >= FEED_MIN_SECONDS:
        entries = feedparser.parse(_get(url).content).entries
        _feeds[url] = (time.time(), entries)
    return entries


def _page(url: str):
    """(publish time or None, text) of an article page, read once. Paywalled pages give no text."""
    if url not in _pages:
        time.sleep(PAGE_DELAY_SECONDS)
        page = _get(url).content
        head = page[:400_000].decode("utf-8", "replace")
        found = (re.search(r'article:published_time"\s+content="([^"]+)"', head)
                 or re.search(r'"datePublished"\s*:\s*"([^"]+)"', head))
        free = not re.search(r'"isAccessibleForFree"\s*:\s*"?false', head, re.I)
        text = trafilatura.extract(page, include_comments=False, include_tables=False) if free else None
        _pages[url] = (_utc(found.group(1)) if found else None, plain(text or ""))
        if len(_pages) > 5000:  # forget the oldest
            for old in list(_pages)[:1000]:
                del _pages[old]
    return _pages[url]


def _from_feeds(outlet: str, urls: list, mode: str, since: pd.Timestamp) -> list:
    out = []
    for feed_url in urls:
        try:
            entries = _entries(feed_url)
        except Exception as e:
            print(f"  news: {outlet} feed {feed_url} failed ({e}) -- skipped")
            continue
        for e in entries:
            stamp = e.get("published_parsed") or e.get("updated_parsed")
            link = e.get("link")
            if not stamp or not link:
                continue
            published = pd.Timestamp(calendar.timegm(stamp), unit="s", tz="UTC")
            if published <= since:   # the feed's time is never before the publish time
                continue
            body = plain(e.content[0].value if e.get("content") else e.get("summary", ""))
            if mode == "page":
                try:
                    page_time, page_text = _page(link)
                    published, body = page_time or published, page_text or body
                except Exception as ex:
                    print(f"  news: {outlet} page {link} failed ({ex}) -- using the feed's summary")
                if published <= since:
                    continue
            out.append(dict(uri=f"{_key(outlet)}:{link}", title=one_line(e.get("title", "")), body=FOOTER.sub("", body),
                            source=outlet, url=link, symbols=[], published=published))
    return out


# --- Alpaca ---

def _from_alpaca(since: pd.Timestamp) -> list:
    global _alpaca
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.requests import NewsRequest
    _alpaca = _alpaca or NewsClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    news = _alpaca.get_news(NewsRequest(start=since.to_pydatetime(), limit=ALPACA_MAX_ARTICLES,
                                        include_content=True, sort="desc")).data.get("news", [])
    out = []
    for n in news:
        published = _utc(n.created_at)
        if published > since:
            out.append(dict(uri=f"alpaca:{n.id}", title=one_line(n.headline), body=plain(n.content or n.summary or ""),
                            source="Benzinga" if n.source == "benzinga" else n.source, url=n.url or "",
                            symbols=list(n.symbols or []), published=published))
    return out


# --- Everything ---

def fetch_new(since: pd.Timestamp) -> list[dict]:
    """Articles published after `since` (tz-aware UTC) from every source in
    config.LIVE_NEWS_SOURCES, newest last. Each: uri, title, body, source, url,
    symbols, published. The same headline from two sources is kept once (the earliest)."""
    since = _utc(since)
    articles = []
    for name in config.LIVE_NEWS_SOURCES:
        try:
            if name == "Alpaca":
                articles += _from_alpaca(since)
            elif name in FEEDS:
                articles += _from_feeds(name, *FEEDS[name], since)
            else:
                print(f"  news: unknown source {name!r} in LIVE_NEWS_SOURCES -- skipped")
        except Exception as e:
            print(f"  news: {name} failed ({e}) -- skipped")
    seen, out = set(), []
    for a in sorted(articles, key=lambda a: a["published"]):
        title = re.sub(r"\W+", " ", a["title"].lower()).strip()
        if a["uri"] in seen or (title and title in seen):   # e.g. one CNBC story in two CNBC feeds
            continue
        seen.update((a["uri"], title))
        out.append(a)
    return out


if __name__ == "__main__":
    now = pd.Timestamp.now(tz="UTC")
    items = fetch_new(now - pd.Timedelta(hours=3))
    df = pd.DataFrame(items)
    if df.empty:
        raise SystemExit("No articles.")
    df["length"] = df["body"].str.len()
    summary = df.groupby("source").agg(articles=("uri", "size"), newest=("published", "max"),
                                       median_body=("length", "median"))
    summary["newest_min_ago"] = ((now - summary.pop("newest")).dt.total_seconds() / 60).round(1)
    print(f"Now {now:%Y-%m-%d %H:%M} UTC, articles from the last 3 hours:")
    print(summary.sort_values("articles", ascending=False).to_string())
