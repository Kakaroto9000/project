"""
Job 2: Claude (TAGGING_MODEL) reads each news article and lists the tickers of the
companies it affects, each with a role.

Two commands, because Claude's Batch API (half the price of normal calls) works in the
background, usually within an hour, at most 24h:
  job2-tag-submit   picks the articles waiting in news_raw.json and sends them in
                    batches of TAGGING_BATCH_SIZE; batch IDs go to tag_batch_state.json.
  job2-tag-collect  checks the batches; re-run it until they're done. Then adds the
                    results to news_tagged.json and removes the articles from news_raw.json.

Skipped: pages that aren't news (NOT_ARTICLE_URLS) and repeat URLs (an article saved twice
under one URL, even one already tagged earlier, keeps its earliest copy). Other outlets'
reports of the same story are all tagged; dataset.py decides whether training uses them
(DATASET_SKIP_STORY_REPEATS).

Each article's headline + body (cut to ARTICLE_CHARS_FOR_TAGGING if set) is sent;
Claude returns up to MAX_COMPANIES_PER_ARTICLE tickers. Tickers aren't checked here --
prices.py finds out which ones are real. dataset.py makes one row per ticker.

Cost: ~$130 for ~195k articles at batch prices with the prompt cached (measured 2026-10-06).
Needs ANTHROPIC_API_KEY and credits on the Anthropic account.
"""

import json
from datetime import datetime
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from newsmodel import config

STATE_PATH = config.RAW_NEWS_DIR / "tag_batch_state.json"  # exists only while batches are out

# This is the instruction Claude gets for every article -- edit it to change what counts
# as "affected". Keep TICKER_SCHEMA and ROLES in sync if you change the output shape
# (parse_answer() expects {"companies": [{"ticker": "TSLA", "role": "direct"}, ...]}).
# The rules under "Tickers" and the examples target mistakes seen in a 200-article test.
TAGGING_SYSTEM_PROMPT = f"""You are tagging news for a stock-direction prediction pipeline. Articles can be about anything: business and finance, technology and science, politics, conflicts, disasters, health, energy, or world events.

Given a news article's headline and body text, identify which publicly-traded companies' stock prices are likely to move materially because of the news described -- up or down.

Include a company only if a trader reading this article would expect its stock to move noticeably in the next day or two because of it. With that test, think beyond the companies the article names, and give each company the role that fits best:
- "direct": a company the article is actually about -- its own results, deal, product, lawsuit, executive or approval. A company that only moves along with its sector or the market is not "direct".
- "competitor": one company's win (a contract, an approval, a product launch) can hurt its rivals, and one company's trouble (a recall, a scandal, an outage) can help them -- or drag the whole sector down with it.
- "opposite_market": a breakthrough in one technology can hurt the companies whose business it threatens (e.g. better batteries or cheaper solar hurt oil and gas; a new drug hurts makers of the treatment it replaces) -- but only when the news is big enough to move that market now (a launch, a deal, an approval, a large-scale result), not early-stage research. Not for price moves: when oil, gas or metal prices move, their producers and big buyers are "supplier_customer".
- "supplier_customer": suppliers and customers whose sales or costs clearly depend on the news (e.g. a chip shortage hits automakers; an oil price jump hits airlines and lifts oil producers).
- "world_event": companies exposed to a war, sanctions, tariffs, an election, a disaster or an epidemic in a region they operate in, sell to or source from.
- "policy_macro": companies most sensitive to an interest-rate decision, inflation or jobs data, a new regulation, a tax change or a court ruling for a whole industry (e.g. a rate cut lifts homebuilders and regional banks; a drug-pricing rule hits pharma).
- "other": a clear, material effect that fits none of the roles above.

Do not include:
- companies merely mentioned in passing or for context, e.g. a bank or analyst quoted only as the source of a forecast;
- companies that only provide a technology the story mentions in general terms
(a story about governments using AI does not move big tech) -- but do include
them when the story names them or involves a concrete contract, purchase,
spending plan or ban;
- companies whose link to the news is speculative or very indirect.

Tickers:
- Only symbols traded on a US exchange (NYSE or NASDAQ): 1-5 capital letters, plus a share class where the company has one (e.g. "BRK.B", "BF.B"). Never numbers or foreign exchange suffixes ("9502", "2222.HK", "MPS.MI").
- Funds: when the article reports a significant market-moving event for a commodity, a sector, interest rates or the whole market -- a big price move, a rate decision, a data release or policy change that moved markets -- you may add up to 2 well-known US-listed exchange-traded funds that track it most directly, with role "fund" (e.g. oil "USO", gold "GLD", long-term Treasuries "TLT", the S&P 500 "SPY", a sector fund such as "XLE" or "SMH"). No funds for market previews or "little changed" days, opinion and commentary pieces, or articles mainly about one company. For such commodity or market-wide news, list at most 5 companies besides the funds, or none.
- A ticker is a short symbol, never a company name ("RENESAS", "WAYMO" and "ARMCO" are not tickers).
- A foreign company counts only through its US listing (ADR, e.g. "TSM", "SFTBY"). If it has none, leave it out -- do not invent one.
- Make sure the ticker belongs to that exact company; similar names are a trap (Shoppers Stop is not "SHOP", which is Shopify).
- Never list private companies (e.g. Waymo, OpenAI).
- If you are not confident of a ticker, leave the company out rather than guessing.

List only the companies you judge genuinely important, ordered from most to least affected. {config.MAX_COMPANIES_PER_ARTICLE} is a hard ceiling, not a target: most articles affect only a few companies (often 1-10), and listing weak or doubtful ones makes the data worse. Only a broad event (e.g. a market-wide or sector-wide one) should come near the ceiling; if more than {config.MAX_COMPANIES_PER_ARTICLE} are affected, keep the top {config.MAX_COMPANIES_PER_ARTICLE}.

If no publicly-traded company is likely to move materially, return an empty list."""

# Worked examples, appended to the prompt. They show the judgment calls (roles, what to
# leave out, the 5-company limit, funds, foreign listings) better than rules alone. The
# companies are invented (tickers "ZZ..."): with real ones the model copied them onto
# unrelated articles (tested 2026-10-06). parse_answer() drops any ZZ ticker that slips through. They also
# bring the prompt past 4,096 tokens, the minimum Claude Haiku caches: cached prompt
# tokens cost 10% of the normal price. Keep them varied (industries, 0/1/several
# tickers) so the model doesn't drift toward any one company or answer size.
TAGGING_EXAMPLES = """
EXAMPLES

These show the judgment and the answer format. The companies in them are invented (their tickers start with "ZZ"), so never use those names or tickers; the funds (USO, TLT, GLD, SPY) are real. Tag only what each new article itself supports. Many articles correctly get an empty list.

Example 1
Article: Pacific Crest Airlines flight forced to land after door panel blows out; regulator grounds 171 Aerion A9 jets
A panel blew out of a Pacific Crest Airlines Aerion A9 shortly after takeoff on Friday. The aviation regulator ordered 171 A9 planes grounded for inspections, and Pacific Crest cancelled hundreds of flights. Unity Air, the largest operator of the A9, also pulled its jets. The fuselage, including the panel, is built by Fusecraft.
Answer: {"companies": [{"ticker": "ZZAR", "role": "direct"}, {"ticker": "ZZFC", "role": "supplier_customer"}, {"ticker": "ZZPC", "role": "direct"}, {"ticker": "ZZUA", "role": "supplier_customer"}]}
Why: Aerion's plane and Pacific Crest's flight are what the story is about; Fusecraft built the failed part; Unity Air loses flights as Aerion's customer.

Example 2
Article: How to retire with $1 million: 5 tips from financial advisors
Advisors recommend maxing out a 401(k), keeping an emergency fund, delaying Social Security and choosing low-cost index funds from large fund providers.
Answer: {"companies": []}
Why: personal-finance advice moves no stock or fund.

Example 3
Article: Regulator approves Halvern Pharma's weight-loss drug
Halvern Pharma won approval for its once-weekly obesity shot, setting up direct competition with Norrland Bio's best-selling drug. Halvern said its shot will cost about 20% less.
Answer: {"companies": [{"ticker": "ZZHV", "role": "direct"}, {"ticker": "ZZNB", "role": "competitor"}]}
Why: Halvern's approval; Norrland (a foreign company with a US listing) faces a cheaper rival.

Example 4
Article: Halvern's obesity drug sharply reduces sleep apnea in late-stage trials
Patients on the drug saw up to a 63% reduction in breathing interruptions during sleep, Halvern said, and it will seek approval for the condition this year. Sleep apnea is today treated mostly with breathing machines from Restwell Medical and implants from Inspira Devices.
Answer: {"companies": [{"ticker": "ZZHV", "role": "direct"}, {"ticker": "ZZRW", "role": "opposite_market"}, {"ticker": "ZZID", "role": "opposite_market"}]}
Why: a large late-stage result that threatens the businesses of the device makers.

Example 5
Article: Fed cuts interest rates by half a percentage point, its first cut in four years
The Federal Reserve lowered its benchmark rate, citing cooling inflation, and signaled more cuts this year. Mortgage rates have already fallen to their lowest level in 18 months.
Answer: {"companies": [{"ticker": "TLT", "role": "fund"}, {"ticker": "ZZBH", "role": "policy_macro"}, {"ticker": "ZZCR", "role": "policy_macro"}]}
Why: a market-wide event: the long-term Treasury fund is the most direct trade, plus a few of the most rate-sensitive companies (homebuilders such as Brightpath Homes and Cedar Ridge Builders).

Example 6
Article: Oil jumps 6% after OPEC+ announces surprise output cut
Brent crude rose above $85 a barrel after OPEC+ members said they would cut production by more than 1 million barrels a day. Analysts at a major bank raised their oil price forecast.
Answer: {"companies": [{"ticker": "USO", "role": "fund"}, {"ticker": "ZZGE", "role": "supplier_customer"}, {"ticker": "ZZPR", "role": "supplier_customer"}, {"ticker": "ZZUA", "role": "supplier_customer"}]}
Why: a commodity price move: the oil fund, producers that gain (Gulfline Energy, Permian Rock Oil) and an airline that pays more for fuel (Unity Air). The bank is only the source of a forecast.

Example 7
Article: Germany's Ostwerk Defence wins record 8.5 billion euro ammunition order
Ostwerk, one of Europe's largest ammunition makers, said the German army ordered artillery shells worth up to 8.5 billion euros. Its shares rose 4% in Frankfurt.
Answer: {"companies": []}
Why: Ostwerk trades only in Europe (no NYSE or NASDAQ listing), and no US-listed company or fund is materially affected.

Example 8
Article: Kestrel Semiconductor profit beats forecasts; chipmaker raises spending plan on AI demand
Kestrel, a foreign chipmaker with US-listed shares, reported a 36% rise in quarterly profit and raised its equipment spending plan. It said demand for AI chips from customers such as Vector Graphics remains extremely strong. Much of its equipment comes from Lumetric Tools.
Answer: {"companies": [{"ticker": "ZZKS", "role": "direct"}, {"ticker": "ZZLT", "role": "supplier_customer"}, {"ticker": "ZZVG", "role": "supplier_customer"}]}
Why: Kestrel's own results; more spending means more orders for its supplier; strong AI demand is read through to its customer.

Example 9
Article: Major bank raises its year-end S&P 500 target
Strategists at a large investment bank lifted their forecast for the benchmark index, citing strong earnings and expected rate cuts.
Answer: {"companies": []}
Why: a forecast is an opinion, not news that moves the market, and the bank is only its source.

Example 9b
Article: Stock futures are little changed as Wall Street awaits Fed minutes
Futures tied to the S&P 500 were flat on Tuesday evening. Investors are waiting for minutes of the Fed's last meeting and for retail earnings later in the week.
Answer: {"companies": []}
Why: a market preview with no move and no event yet: no fund and no company.

Example 10
Article: Attacks push shipping lines to avoid the Red Sea, sending freight rates soaring
Container lines are rerouting ships around Africa, adding up to two weeks to voyages, and rates from Asia to Europe have more than doubled. Meridian Shipping, a US-listed container line, said it is raising prices; Denmark's Nordhavn Line, listed only in Copenhagen, is rerouting too.
Answer: {"companies": [{"ticker": "ZZMS", "role": "world_event"}]}
Why: higher freight rates help the US-listed container line; Nordhavn has no US listing, so it is left out.

Example 11
Article: Defense Department awards $9 billion cloud contract to Stratus Cloud and Orbis Data
The department split its cloud contract between the two companies, which will compete for task orders through 2028.
Answer: {"companies": [{"ticker": "ZZOD", "role": "direct"}, {"ticker": "ZZSC", "role": "direct"}]}
Why: a concrete, named contract; Orbis first, as the contract matters most relative to its size. Compare: a city starting to use AI chatbots on its phone line would get an empty list.

Example 12
Article: Ridgeway Motors recalls 1.9 million SUVs over trim that can fly off
Ridgeway is recalling nearly 1.9 million SUVs because a trim piece can detach while driving. Dealers will replace the clips for free.
Answer: {"companies": [{"ticker": "ZZRM", "role": "direct"}]}
Why: Ridgeway's own recall; a cheap fix that doesn't help or hurt rivals, so no competitors.

Example 13
Article: Corvex Networks to buy Sentinel Analytics for $28 billion in its largest-ever deal
Corvex agreed to pay $157 a share in cash, a 31% premium to Sentinel's last closing price.
Answer: {"companies": [{"ticker": "ZZSA", "role": "direct"}, {"ticker": "ZZCN", "role": "direct"}]}
Why: the target first, since a takeover premium moves it most; the buyer is also directly involved.

Example 14
Article: US quadruples tariffs on Chinese electric vehicles to 100%
The White House raised tariffs on China-made EVs and on Chinese batteries and chips. Jinlong EV, whose shares trade in New York, and Huaxia Motors, listed only in Hong Kong, both sell few cars in the US today.
Answer: {"companies": [{"ticker": "ZZJL", "role": "world_event"}]}
Why: the US-listed Chinese EV maker is directly exposed to the tariff; Huaxia has no NYSE or NASDAQ listing.

Example 15
Article: Greenleaf Pharmacy slashes profit outlook as pharmacy business weakens; shares plunge 20%
Greenleaf cut its full-year earnings forecast, blaming weak consumer spending and lower drug reimbursement rates, and said it will close many US stores.
Answer: {"companies": [{"ticker": "ZZGP", "role": "direct"}, {"ticker": "ZZMD", "role": "competitor"}]}
Why: Greenleaf's own outlook; lower reimbursement rates weigh on its closest rival, Medway Drug, too.

Example 16
Article: Faulty Shieldline Security update crashes millions of computers worldwide
A defective update from cybersecurity firm Shieldline crashed about 8.5 million PCs, disrupting banks and hospitals. Pacific Crest Airlines cancelled thousands of flights over several days. Rival Bastion Cyber said customers had started asking about switching.
Answer: {"companies": [{"ticker": "ZZSL", "role": "direct"}, {"ticker": "ZZPC", "role": "supplier_customer"}, {"ticker": "ZZBC", "role": "competitor"}]}
Why: Shieldline's own failure; the airline was hit hardest as a customer; a rival may win customers.

Example 17
Article: Hurricane makes landfall in Florida as a Category 3 storm
The storm knocked out power to more than 3 million homes. Early estimates put insured losses at $30 billion to $50 billion.
Answer: {"companies": [{"ticker": "ZZSI", "role": "world_event"}, {"ticker": "ZZBR", "role": "world_event"}]}
Why: a Florida-focused home insurer (Sunshore Insurance) faces claims; a home-improvement chain (BuildRight Stores) sees rebuilding demand. A few of the most exposed, not every insurer.

Example 18
Article: Ukrainian forces retake villages near Kharkiv after weeks of fighting
Ukraine's army said it had pushed Russian troops back from several villages in the northeast. Fighting continues along the front line.
Answer: {"companies": []}
Why: battlefield news with no company, contract or market directly involved.

Example 19
Article: Judge rules Searchwise illegally monopolized online search
A federal judge found that Searchwise broke antitrust law, in part by paying billions of dollars a year to be the default search engine on phones. It paid Pomelo Devices about $20 billion last year for default placement. Remedies will be decided later.
Answer: {"companies": [{"ticker": "ZZSW", "role": "direct"}, {"ticker": "ZZPD", "role": "supplier_customer"}]}
Why: the ruling is against Searchwise; Pomelo's large payments from it are at risk.

Example 20
Article: Stride Athletic upgraded to buy by analysts who see turnaround under new CEO
Analysts at a major bank raised their rating on the sneaker maker, saying new products should revive sales growth next year.
Answer: {"companies": [{"ticker": "ZZST", "role": "direct"}]}
Why: a rating change is news about the rated company; the bank that issued it is not affected.

Example 21
Article: Scientists build a battery that charges in five minutes in lab tests
University researchers said a new electrode material let a small test cell charge fully in five minutes. They said commercial use is likely at least a decade away.
Answer: {"companies": []}
Why: early-stage research, not big enough to move battery makers or automakers now.

Example 22
Article: Cocoa prices hit a record as West African harvests fail
Cocoa futures have more than doubled this year after poor weather and crop disease. Chocolate makers Cacao Crown and Sweetfield Foods have warned of higher costs and are raising prices.
Answer: {"companies": [{"ticker": "ZZCC", "role": "supplier_customer"}, {"ticker": "ZZSF", "role": "supplier_customer"}]}
Why: a commodity price move hurts the companies that buy the most of it. No well-known US fund tracks cocoa, so none is listed.

Example 23
Article: Gold hits a record high as investors seek safety
Gold rose above $2,500 an ounce for the first time as expectations of rate cuts and geopolitical worries drove demand for safe assets.
Answer: {"companies": [{"ticker": "GLD", "role": "fund"}]}
Why: a commodity move with no single company at its center: the gold fund is the direct trade.

Example 24
Article: Stocks slide as weak jobs report stokes recession fears
US employers added far fewer jobs than expected last month and unemployment rose to a three-year high. The S&P 500 fell 2% and Treasury yields dropped sharply.
Answer: {"companies": [{"ticker": "SPY", "role": "fund"}, {"ticker": "TLT", "role": "fund"}]}
Why: a market-wide move: the S&P 500 fund and the long-term Treasury fund, rather than a list of individual stocks.
"""

TAGGING_SYSTEM_PROMPT += "\n" + TAGGING_EXAMPLES

ROLES = ["direct", "competitor", "opposite_market", "supplier_customer", "world_event", "policy_macro", "other",
         "fund"]  # "fund": an ETF for commodity, sector, rate or market-wide news

# The exact JSON shape Claude must answer in. Claude's structured outputs don't accept
# "maxItems", so the MAX_COMPANIES_PER_ARTICLE ceiling is in the prompt and in parse_answer().
TICKER_SCHEMA = {
    "type": "object",
    "properties": {
        "companies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "role": {"type": "string", "enum": ROLES},  # only these values are allowed
                },
                "required": ["ticker", "role"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["companies"],
    "additionalProperties": False,
}

# Enough room for a full MAX_COMPANIES_PER_ARTICLE response (~15 tokens per company);
# a response cut off mid-JSON would be unparseable and lose all of that article's tickers.
TAGGING_MAX_TOKENS = 100 + config.MAX_COMPANIES_PER_ARTICLE * 15

# Pages Event Registry saves as articles that aren't news, skipped by URL.
NOT_ARTICLE_URLS = [
    "reuters.com/markets/companies/",  # stock-quote pages: company description + latest-story teaser
]


# --- Files ---

def read_articles(path: Path) -> list:
    """Articles saved in `path`; an empty list if the file doesn't exist yet."""
    return json.loads(path.read_text()) if path.exists() else []


def write_articles(path: Path, articles: list):
    """Save `articles` to `path`. Written to a temporary file first and then swapped in,
    so a crash mid-write can't corrupt the existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".json.tmp")
    with open(temp_path, "w") as f:
        json.dump(articles, f)
    temp_path.replace(path)  # one step: the file is either all old or all new


# --- job2-tag-submit ---

def select_articles(articles: list, tagged: list = ()) -> list:
    """Articles to send, oldest first. Skips non-news pages (NOT_ARTICLE_URLS) and any URL
    that's already tagged or appeared earlier, so each URL keeps its first copy."""
    seen_urls = {a.get("url") for a in tagged}
    selected = []
    for article in sorted(articles, key=lambda a: a.get("dateTime") or "9999"):  # no date -> last
        url = article.get("url") or ""
        if url in seen_urls or any(part in url for part in NOT_ARTICLE_URLS):
            continue
        seen_urls.add(url)
        selected.append(article)
    return selected


def build_request(i: int, item: dict) -> Request:
    """One Claude request for one article: the instructions, headline + start of the body,
    and the answer format. custom_id "item-<i>" matches the answer back to the article."""
    body = (item.get("body") or "")[:config.ARTICLE_CHARS_FOR_TAGGING]
    text = f"{item.get('title', '')}\n\n{body}".strip()
    return Request(
        custom_id=f"item-{i}",  # i = position in the pending file
        params=MessageCreateParamsNonStreaming(
            model=config.TAGGING_MODEL,
            max_tokens=TAGGING_MAX_TOKENS,
            # Cached: identical for every article. 1-hour lifetime because batch requests can be
            # minutes apart; reads cost 10% of the normal price, the first write 2x.
            system=[{"type": "text", "text": TAGGING_SYSTEM_PROMPT,
                     "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
            messages=[{"role": "user", "content": text}],
            output_config={"format": {"type": "json_schema", "schema": TICKER_SCHEMA}},
        ),
    )


def submit_batches(client: anthropic.Anthropic, pending: list, handled_uris: list, max_batches: int = None):
    """Send `pending` in batches of TAGGING_BATCH_SIZE and record everything collect() needs
    in tag_batch_state.json. `handled_uris` (every article looked at, sent or skipped) is
    removed from news_raw.json once the results are in."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pending_path = config.RAW_NEWS_DIR / f"tag_pending_{timestamp}.json"
    with open(pending_path, "w") as f:
        json.dump(pending, f)  # answers come back as "item-<i>"; this file says which article i is

    state = {"pending_file": str(pending_path), "handled_uris": handled_uris, "batch_ids": [], "sent": 0}
    for start in range(0, len(pending), config.TAGGING_BATCH_SIZE):
        end = min(start + config.TAGGING_BATCH_SIZE, len(pending))  # last batch may be smaller
        requests = [build_request(i, pending[i]) for i in range(start, end)]
        batch = client.messages.batches.create(requests=requests)  # returns once uploaded
        state["batch_ids"].append(batch.id)
        state["sent"] = end  # pending[:sent] went out; the rest stays in news_raw.json if we stop here
        with open(STATE_PATH, "w") as f:
            json.dump(state, f)  # after every batch, so a failure partway keeps the paid-for ones
        print(f"  submitted batch {batch.id}: articles {start}-{end - 1}", flush=True)
        if max_batches and len(state["batch_ids"]) == max_batches and end < len(pending):
            print(f"  stopping after {max_batches} batches; {len(pending) - end} articles stay "
                  f"in {config.RAW_NEWS_PATH.name} for the next submit")
            return


def submit(max_batches: int = None):
    """job2-tag-submit: pick the articles waiting in news_raw.json and send them to Claude,
    at most `max_batches` batches if given (the rest wait for the next submit).
    Refuses to run while earlier batches haven't been collected."""
    if STATE_PATH.exists():
        raise SystemExit("Batches were already sent and not collected yet -- run job2-tag-collect.")
    client = anthropic.Anthropic(timeout=1800.0)  # 30 min: each batch is a large upload

    raw = read_articles(config.RAW_NEWS_PATH)
    pending = select_articles(raw, read_articles(config.TAGGED_NEWS_PATH))
    print(f"{len(raw)} articles waiting -> {len(pending)} to tag "
          f"(repeat URLs and non-article pages skipped)")
    if not pending:
        print("Nothing new to tag.")
        return

    submit_batches(client, pending, [a["uri"] for a in raw], max_batches)
    print("Batches usually finish within an hour (max 24h) -- run job2-tag-collect later.")


# --- job2-tag-collect ---

def batches_done(client: anthropic.Anthropic, batch_ids: list) -> bool:
    """Print each batch's progress; True once every batch has ended."""
    batches = [client.messages.batches.retrieve(batch_id) for batch_id in batch_ids]
    for batch in batches:
        print(f"Batch {batch.id}: {batch.processing_status} {batch.request_counts}")
    return all(batch.processing_status == "ended" for batch in batches)


def parse_answer(text: str) -> tuple:
    """(tickers in Claude's order, {ticker: role}) from its JSON answer, at most
    MAX_COMPANIES_PER_ARTICLE; empty if the answer can't be read."""
    try:
        companies = json.loads(text).get("companies", [])
    except (json.JSONDecodeError, AttributeError):
        return [], {}
    roles = {}
    for company in companies:
        ticker = str(company.get("ticker", "")).strip().upper()
        if ticker and not ticker.startswith("ZZ"):  # ZZ = invented companies from the examples
            roles.setdefault(ticker, company.get("role", "other"))  # a repeat keeps its first role
        if len(roles) == config.MAX_COMPANIES_PER_ARTICLE:
            break
    return list(roles), roles


def collect_results(client: anthropic.Anthropic, batch_ids: list, pending: list) -> tuple:
    """(tagged, failed): the pending articles Claude answered, each with "tickers" (most
    affected first) and "ticker_roles" ({ticker: role}) -- an unreadable answer leaves both
    empty -- and the articles whose request failed (errored, expired, canceled, e.g. when the
    account ran out of credits), which should be sent again."""
    answered = {}
    for batch_id in batch_ids:
        for result in client.messages.batches.results(batch_id):
            i = int(result.custom_id.split("-", 1)[1])  # "item-42" -> 42
            if result.result.type != "succeeded":
                continue
            msg = result.result.message
            text = next((b.text for b in msg.content if b.type == "text"), "{}")
            tickers, roles = parse_answer(text)
            answered[i] = dict(pending[i], tickers=tickers, ticker_roles=roles)
    failed = [item for i, item in enumerate(pending) if i not in answered]
    return list(answered.values()), failed


def save_results(tagged: list, handled_uris: list):
    """Add the tagged articles to news_tagged.json, then remove every handled article from
    news_raw.json. In that order: if the run stops in between, collecting again is safe."""
    already = read_articles(config.TAGGED_NEWS_PATH)
    already_uris = {a["uri"] for a in already}
    new = [a for a in tagged if a["uri"] not in already_uris]  # a re-collect doesn't add twice
    write_articles(config.TAGGED_NEWS_PATH, already + new)

    handled = set(handled_uris)
    waiting = [a for a in read_articles(config.RAW_NEWS_PATH) if a["uri"] not in handled]
    write_articles(config.RAW_NEWS_PATH, waiting)  # keeps articles Job 1 added in the meantime
    print(f"Added {len(new)} to {config.TAGGED_NEWS_PATH.name}, "
          f"removed {len(handled)} from {config.RAW_NEWS_PATH.name}")


def collect():
    """job2-tag-collect: if every batch is done, save the results;
    otherwise print the progress so you can run it again later."""
    if not STATE_PATH.exists():
        raise SystemExit("No batches waiting -- run job2-tag-submit first.")
    client = anthropic.Anthropic()

    state = json.loads(STATE_PATH.read_text())
    if not batches_done(client, state["batch_ids"]):
        print("Not all batches are done -- run job2-tag-collect again later.")
        return

    pending = json.loads(Path(state["pending_file"]).read_text())
    sent, unsent = pending[:state["sent"]], pending[state["sent"]:]
    if unsent:  # submit stopped partway: these were never sent, so they stay waiting
        print(f"{len(unsent)} articles were never sent (submit stopped partway); "
              f"they stay in {config.RAW_NEWS_PATH.name} for the next submit")
    tagged, failed = collect_results(client, state["batch_ids"], sent)
    with_tickers = sum(1 for a in tagged if a["tickers"])
    print(f"{with_tickers}/{len(tagged)} articles tagged with at least one ticker")
    if failed:
        print(f"{len(failed)} requests failed (e.g. no credits left); those articles stay in "
              f"{config.RAW_NEWS_PATH.name} for the next submit")
    keep_waiting = {a["uri"] for a in unsent + failed}
    save_results(tagged, [uri for uri in state["handled_uris"] if uri not in keep_waiting])

    STATE_PATH.unlink()  # nothing is out anymore -> next run is a submit
    Path(state["pending_file"]).unlink()
