"""
Podcast Contextual Matching Engine
==================================

Finds recent medical news articles that are topically close to episodes in
your podcast back catalogue, then drafts native-ad headlines for the best
matches. Designed to run once a day (e.g. via cron) and write out
matches.json, which dashboard.html reads.

WHAT THIS DOES
1. Loads your episode transcripts + builds (and caches) embeddings for them.
2. Pulls recent articles from a list of RSS feeds (medical publishers /
   news sources).
3. Embeds each article's title+summary and compares it to every episode
   via cosine similarity.
4. Keeps matches above a similarity threshold.
5. For the strongest matches, drafts a few native-ad headlines in the
   "Continue learning" style from the AudienceLift model — free, from fixed
   templates by default; optionally via the Claude API if ANTHROPIC_API_KEY
   is set (costs money, entirely optional, see draft_headlines() below).
6. Writes everything to matches.json for the dashboard.

SETUP
    pip install feedparser scikit-learn anthropic
    pip install sentence-transformers   # optional but recommended — see below

    No API key needed for normal use — see draft_headlines() below.

    Put one text file per episode in ./episodes/, named however you like,
    with this format (first line = title, second line = url, rest = transcript):

        Should every diabetic now receive SGLT2 inhibitors?
        https://yourpodcast.com/episodes/sglt2-inhibitors
        [transcript text...]

RUNNING
    python matching_engine.py

    Then schedule it, e.g. with cron:
        0 6 * * * cd /path/to/podcast_matching && /usr/bin/python3 matching_engine.py

EMBEDDING BACKEND
This script prefers sentence-transformers (semantic, paraphrase-aware
matching) and automatically falls back to a local TF-IDF + cosine-similarity
matcher if sentence-transformers/torch isn't installed. TF-IDF only catches
keyword/lexical overlap, not paraphrases, and produces noticeably lower
similarity scores — so SIMILARITY_THRESHOLD may need separate tuning for
each backend. The script prints which backend is active and the full score
table every run so you can see what you're tuning against. Install
sentence-transformers for meaningfully better matching if your machine has
the disk space and a normal internet connection (the model + torch is
roughly 500MB-1GB combined).

RSS FEEDS
The feed list below was verified live on 2026-08-05 by fetching each site
directly. See the comment above RSS_FEEDS for exactly what was and wasn't
confirmed, and which publishers are worth emailing directly for a partner
feed instead of scraping.
"""

import os
import sys
import json
import glob
import pickle
import hashlib
import socket
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone

import feedparser
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# CONFIG — edit this section for your show
# ---------------------------------------------------------------------------

EPISODES_DIR = "episodes"          # one .txt file per episode
EPISODE_INDEX_FILE = "episode_index.json"  # cloud-mode fallback — see
                                    # export_episode_index.py. Contains only
                                    # episode titles, URLs and pre-computed
                                    # embedding vectors (numbers), never the
                                    # raw transcript text. Used automatically
                                    # when ./episodes/ isn't present (e.g. on
                                    # a public GitHub Actions runner, where
                                    # the full transcripts deliberately
                                    # aren't checked in).
CACHE_FILE = "episode_embeddings.pkl"
OUTPUT_FILE = "matches.json"
OFFLINE_ARTICLES_CACHE = "sample_articles_cache.json"  # see README — used
                                                        # only if live RSS
                                                        # fetching returns
                                                        # zero articles

SIMILARITY_THRESHOLD = 0.4         # Lowered from 0.45 on 2026-08-06 after the
                                    # first sentence-transformers run: 0.45
                                    # only surfaced 1 match (a good one) —
                                    # loosening slightly to let a few more
                                    # through while still well clear of the
                                    # noise floor. Keep watching the "Score
                                    # table" printed each run; nudge up if
                                    # weak/off-topic matches start slipping
                                    # in, nudge down further if it's still
                                    # too sparse after a few more real runs.
                                    # If you ever go back to TF-IDF, drop
                                    # this to ~0.05 instead — the two
                                    # backends' score distributions are not
                                    # comparable.
FEED_SIMILARITY_THRESHOLD = 0.45   # Threshold for episodes embedded from
                                    # their RSS title + show notes (see
                                    # build_feed_index.py) rather than a full
                                    # transcript. Set 2026-11-23 after scoring
                                    # 86 real articles against the full
                                    # 141-episode index: at 0.40 over half of
                                    # all articles "matched", mostly via
                                    # generic episodes (net-zero health system,
                                    # panel discussions) pairing with health-
                                    # policy news; at 0.45 the show-notes
                                    # matches left were mostly on-topic (oral
                                    # GLP-1s, ADHD, sleep, dementia blood test)
                                    # and the overall match rate (~30% of
                                    # articles) is close to the old 25-episode
                                    # index's (~23%). Transcript-based episodes
                                    # keep SIMILARITY_THRESHOLD above.
# Recency boost — favour newer episodes in close calls (added 2026-11-23).
# Among the episodes that qualify for an article, the winner is the one with
# the highest
#     adjusted = similarity + RECENCY_MAX_BOOST * 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)
# so a just-released episode gets +0.04, a 6-month-old one ~+0.02, a
# year-old one ~+0.01, an undated one nothing. An older episode that is a
# clearly better fit (by more than the bonus gap, at most ~0.03) still wins.
#
# The threshold uses RAW similarity (RECENCY_THRESHOLD_ALLOWANCE = 0): the
# bonus decides WHICH episode an article gets, never WHETHER it gets one.
# Tested on 86 real articles (see PR): an allowance of 0.02 (raw may fall
# 0.02 short if the bonus lifts it over) let in 5 extra matches and all were
# weak — e.g. a nuclear-weapons story -> "Delivering a net-zero health
# system", a GP-politics op-ed -> "Engaging Men in Primary Care". Raise it
# only if you want the bonus to admit borderline matches too.
RECENCY_MAX_BOOST = 0.04
RECENCY_HALF_LIFE_DAYS = 180
RECENCY_THRESHOLD_ALLOWANCE = 0.0

TOP_MATCHES_PER_ARTICLE = 1        # how many episodes to attach per article
LOOKBACK_HOURS = 36                # how far back to scan RSS feeds
HEADLINES_PER_MATCH = 3            # ad headline variants to draft per match
FEED_TIMEOUT_SECONDS = 10          # max time to wait on any one RSS feed
                                    # before giving up and moving on — without
                                    # this, a single slow/unresponsive feed
                                    # can hang the whole run indefinitely
                                    # (feedparser has no timeout by default)

# Verified live on 2026-08-05 by fetching each site directly:
#   - newsGP: confirmed via the site's own RSS autodiscovery link
#     (?rss=RACGPnewsGPArticles)
#   - Medical Republic: confirmed, /feed/ returns application/rss+xml
#   - Croakey: confirmed, /feed/ returns application/rss+xml
# Checked and found NOT to have a public feed (2026-08-05):
#   - AusDoc: /feed/ redirects straight to a login page — content is
#     account-gated. Worth emailing their ad sales/partnerships team for a
#     direct feed instead.
#   - Healthed: React/Elementor-based site, no RSS autodiscovery link in the
#     page source, no working /feed/ path found.
#   - MJA (mja.com.au) and TGA/health.gov.au: no working feed found at the
#     usual paths — may need a different URL or may not expose one publicly.
#   - GPonline (UK), NZ Doctor, HSE (UK), NZ Herald, HSJ (UK): checked, no
#     working feed found at the standard paths.
#
# Added 2026-08-05, also verified live by fetching directly — broader than
# pure medical news, on request:
#   - Safe Work Australia: confirmed, /rss.xml returns real content (already
#     surfaced an "occupational lung diseases in Australia" report — a
#     strong topical fit for the silicosis episode)
#   - Pulse Today: confirmed, /feed/ returns application/rss+xml. This is
#     the UK's equivalent of newsGP — GP-specific political, financial and
#     clinical news. Worth noting: it's a UK publication, so a "match" here
#     is topically relevant but the ad would be serving a UK audience, not
#     necessarily one your dad's AU-GP listeners are reading. Worth watching
#     whether it actually produces usable placements or just noise.
#   - WorkSafe NZ: confirmed, /rss.xml returns real content — NZ equivalent
#     of Safe Work Australia, still occupational-health-relevant.
# (BBC News Health and RNZ were also verified live and working, but left
# out on request — general NZ/UK news was judged too far from an
# Australian GP audience to be worth the noise.)
#
# REMOVED 2026-08-06 after a real run surfaced clearly unrelated matches
# ("MODERATE MATCH" against AFL and All Blacks selection stories, and a dog
# behaviour story matching a kidney disease episode):
#   - ABC News Health (feed ID 51120): this ID was carried over from the
#     original placeholder script and never actually verified — the fetch
#     tool used to verify the other feeds blocks abc.net.au by policy, so
#     it went in on trust. Checked the ID afterwards against ABC's actual
#     published topic feed list and 51120 isn't on it; it appears to have
#     been resolving to general/top-story content (hence the AFL match),
#     not health news specifically. If you want ABC health content back,
#     someone needs to find the correct topic contentID from the page
#     source of abc.net.au/news/health (view source, look for the
#     ContentID in the page's <meta> data) and rebuild the URL as
#     abc.net.au/news/feed/<that ID>/rss.xml — don't re-add the old one.
#   - Stuff.co.nz: this was explicitly a "wide net" general NZ news feed
#     (see the note it was added with), not a health-specific one — the
#     rugby selection story and the dog-behaviourist story both came from
#     here. Removing it rather than trying to filter it, since Stuff
#     doesn't appear to expose a health-only RSS feed.
RSS_FEEDS = [
    ("newsGP (RACGP)", "https://www1.racgp.org.au/newsgp?rss=RACGPnewsGPArticles"),
    ("Medical Republic", "https://www.medicalrepublic.com.au/feed/"),
    ("Croakey Health Media", "https://www.croakey.org/feed/"),
    ("Safe Work Australia", "https://www.safeworkaustralia.gov.au/rss.xml"),
    ("Pulse Today (UK)", "https://www.pulsetoday.co.uk/feed/"),
    ("WorkSafe NZ", "https://www.worksafe.govt.nz/about-us/news-and-media/rss"),
]

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small, local, no API key needed

CLAUDE_MODEL = "claude-sonnet-5"   # was "claude-sonnet-4-6" in the original
                                    # script, which is not a real model ID —
                                    # headline drafting was silently failing
                                    # every time. Fixed here.


# ---------------------------------------------------------------------------
# EMBEDDING BACKEND
# ---------------------------------------------------------------------------

def get_embedding_backend():
    """Prefers sentence-transformers; falls back to TF-IDF if unavailable
    (e.g. no torch, no internet/disk budget to install it)."""
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
        return "sentence-transformers"
    except ImportError:
        print("NOTE: sentence-transformers not installed — using a local "
              "TF-IDF embedder instead (keyword/lexical matching only, no "
              "model download required). For semantic, paraphrase-aware "
              "matching, install it with:\n"
              "    pip install sentence-transformers\n")
        return "tfidf"


def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL_NAME)


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_episodes():
    episodes = []
    for path in sorted(glob.glob(os.path.join(EPISODES_DIR, "*.txt"))):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        if len(lines) < 3:
            print(f"Skipping {path}: expected title, url, then transcript")
            continue
        title, url = lines[0].strip(), lines[1].strip()
        transcript = "\n".join(lines[2:]).strip()
        episodes.append({
            "path": path,
            "title": title,
            "url": url,
            "transcript": transcript,
            "source": "transcript",
            "hash": file_hash(path),
        })
    return episodes


def load_episodes_from_index():
    """Cloud-mode loader: reads EPISODE_INDEX_FILE instead of raw transcript
    files. Only title, url and a pre-computed embedding vector are present —
    no transcript text ever touches this path, which is what makes it safe
    to run on a public GitHub Actions runner / public repo. Only works with
    the sentence-transformers backend, since the vectors were computed with
    that model — there's no raw text here for TF-IDF to fit a vocabulary on."""
    import numpy as np
    with open(EPISODE_INDEX_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    episodes = []
    for ep in data.get("episodes", []):
        episodes.append({
            "path": EPISODE_INDEX_FILE,
            "title": ep["title"],
            "url": ep["url"],
            "source": ep.get("source", "transcript"),
            "published": ep.get("published"),
            "embedding": np.array(ep["embedding"], dtype="float32"),
        })
    return episodes


def add_feed_episodes_from_index(local_episodes):
    """Local runs: ./episodes/ only holds the episodes you have transcripts
    for. Add the rest of the catalogue from EPISODE_INDEX_FILE's feed-based
    entries (title + show-notes embeddings, see build_feed_index.py),
    skipping any that a local transcript already covers."""
    if not os.path.exists(EPISODE_INDEX_FILE):
        return local_episodes
    from build_feed_index import same_title
    index_eps = load_episodes_from_index()
    # Local transcripts carry no release date; take it from the index.
    for loc in local_episodes:
        loc.setdefault("published", next(
            (ep["published"] for ep in index_eps
             if ep["url"] == loc["url"] or same_title(ep["title"], loc["title"])),
            None,
        ))
    feed_eps = [
        ep for ep in index_eps
        if ep["source"] == "feed" and not any(
            loc["url"] == ep["url"] or same_title(loc["title"], ep["title"])
            for loc in local_episodes
        )
    ]
    print(f"  + {len(feed_eps)} more episodes from {EPISODE_INDEX_FILE} "
          f"(embedded from podcast feed show notes)")
    return local_episodes + feed_eps


def load_episodes_for_matching(backend):
    """Picks the right episode source automatically:
    - ./episodes/*.txt if present (full local run, either backend) — this is
      untouched, normal behaviour for your own machine.
    - EPISODE_INDEX_FILE if ./episodes/ isn't present (cloud run) — titles,
      URLs and embeddings only, no transcript text. Requires
      sentence-transformers, since that's what the vectors were computed
      with."""
    has_local_transcripts = bool(glob.glob(os.path.join(EPISODES_DIR, "*.txt")))
    if has_local_transcripts:
        return load_episodes(), "local"
        # (feed-based episodes from the index are added in main() once the
        # transcript embeddings are built — sentence-transformers only)

    if os.path.exists(EPISODE_INDEX_FILE):
        if backend != "sentence-transformers":
            print(f"ERROR: found {EPISODE_INDEX_FILE} but no ./{EPISODES_DIR}/ "
                  f"transcripts, and the TF-IDF backend needs the actual "
                  f"transcript text to work (it can't match against "
                  f"pre-computed vectors alone). Install sentence-transformers "
                  f"in this environment, or provide ./{EPISODES_DIR}/.")
            sys.exit(1)
        print(f"No ./{EPISODES_DIR}/ folder found — using {EPISODE_INDEX_FILE} "
              f"instead (titles, URLs and pre-computed embeddings only, no "
              f"transcript text). This is expected in cloud/CI runs.")
        return load_episodes_from_index(), "index"

    print(f"ERROR: no episodes found — need either ./{EPISODES_DIR}/*.txt "
          f"or {EPISODE_INDEX_FILE}.")
    sys.exit(1)


def build_or_load_episode_embeddings(episodes, embedder):
    """sentence-transformers path only — TF-IDF is fit fresh each run in
    match_articles_to_episodes() since it needs the article corpus too."""
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            cache = pickle.load(f)

    changed = False
    for ep in episodes:
        if "embedding" in ep:  # already embedded (index entry)
            continue
        cached = cache.get(ep["path"])
        if cached and cached["hash"] == ep["hash"]:
            ep["embedding"] = cached["embedding"]
            continue
        # Truncate very long transcripts to keep embedding fast/stable —
        # title + first ~2000 words captures the topic well enough for
        # this kind of matching.
        text_for_embedding = ep["title"] + ". " + " ".join(
            ep["transcript"].split()[:2000]
        )
        ep["embedding"] = embedder.encode(text_for_embedding)
        cache[ep["path"]] = {"hash": ep["hash"], "embedding": ep["embedding"]}
        changed = True

    if changed:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(cache, f)

    return episodes


def episode_text_for_matching(ep):
    """Same truncation rule used for both embedding backends, so TF-IDF and
    sentence-transformers see equivalent input."""
    return ep["title"] + ". " + " ".join(ep["transcript"].split()[:2000])


# ---------------------------------------------------------------------------
# NEWS FETCHING
# ---------------------------------------------------------------------------

import re
import html as html_module

_WP_APPEARED_FIRST_ON_RE = re.compile(
    r"<p>\s*The post.*?appeared first on.*?</p>", re.IGNORECASE | re.DOTALL
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_summary(raw, max_len=320):
    """RSS feed 'summary' fields (especially from WordPress sites like
    Medical Republic) often contain raw HTML plus a Jetpack-style 'The post
    X appeared first on Y' trailer. Strip both so matches.json and the
    dashboard show plain, readable text instead of markup."""
    if not raw:
        return ""
    text = _WP_APPEARED_FIRST_ON_RE.sub("", raw)
    text = _HTML_TAG_RE.sub(" ", text)
    text = html_module.unescape(text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if len(text) > max_len:
        truncated = text[:max_len].rsplit(" ", 1)[0]
        text = truncated + "…"
    return text


def fetch_recent_articles():
    cutoff = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)
    articles = []
    for source_name, feed_url in RSS_FEEDS:
        print(f"  Fetching {source_name}...", end=" ", flush=True)
        try:
            # feedparser.parse() has no built-in timeout, so a single slow or
            # unresponsive feed can hang the whole run forever with no error.
            # Fetch the raw bytes ourselves with an explicit timeout, then
            # hand them to feedparser to interpret.
            req = urllib.request.Request(
                feed_url, headers={"User-Agent": "Mozilla/5.0 (compatible; PodcastMatcher/1.0)"}
            )
            with urllib.request.urlopen(req, timeout=FEED_TIMEOUT_SECONDS) as resp:
                raw = resp.read()
            feed = feedparser.parse(raw)
        except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
            print(f"timed out / unreachable after {FEED_TIMEOUT_SECONDS}s — skipping ({e})")
            continue
        except Exception as e:
            print(f"failed — skipping ({e})")
            continue

        if getattr(feed, "bozo", False) and not feed.entries:
            print(f"no parseable entries ({getattr(feed, 'bozo_exception', 'unknown error')})")
        else:
            print(f"{len(feed.entries)} entries")

        for entry in feed.entries:
            published = getattr(entry, "published_parsed", None)
            if published:
                pub_dt = datetime(*published[:6], tzinfo=timezone.utc)
                if pub_dt < cutoff:
                    continue
            else:
                pub_dt = None

            articles.append({
                "source": source_name,
                "title": clean_summary(entry.get("title", ""), max_len=200),
                "summary": clean_summary(entry.get("summary", "")),
                "url": entry.get("link", ""),
                "published": pub_dt.isoformat() if pub_dt else None,
            })
    return articles


def load_offline_articles_cache():
    """Fallback used only when live RSS fetching returns zero articles —
    e.g. running in a network-sandboxed environment. See README."""
    if not os.path.exists(OFFLINE_ARTICLES_CACHE):
        return []
    with open(OFFLINE_ARTICLES_CACHE, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Live RSS fetch returned 0 articles — loading "
          f"{len(data.get('articles', []))} articles from the offline cache "
          f"({OFFLINE_ARTICLES_CACHE}, fetched {data.get('fetched_at', 'unknown date')}). "
          f"This is a real, dated snapshot, not synthetic data — but it will "
          f"go stale. Delete this file once live fetching works from your "
          f"network.")
    return data.get("articles", [])


# ---------------------------------------------------------------------------
# MATCHING
# ---------------------------------------------------------------------------

def match_articles_to_episodes_st(articles, episodes, embedder):
    """sentence-transformers backend: per-item .encode() + cosine similarity,
    same as the original script."""
    if not episodes:
        return []

    ep_embeddings = [ep["embedding"] for ep in episodes]
    matches = []

    for art in articles:
        text = f"{art['title']}. {art['summary']}"
        if not text.strip():
            continue
        art_embedding = embedder.encode(text)

        sims = cosine_similarity([art_embedding], ep_embeddings)[0]
        matches.extend(rank_episodes(art, episodes, sims, use_recency=True))

    matches.sort(key=lambda m: m["adjusted_score"], reverse=True)
    return matches


def threshold_for(match):
    if match["episode"].get("source") == "feed":
        return FEED_SIMILARITY_THRESHOLD
    return SIMILARITY_THRESHOLD


def recency_boost(published, today=None):
    """Bonus for newer episodes: RECENCY_MAX_BOOST on release day, halving
    every RECENCY_HALF_LIFE_DAYS. Undated episodes get nothing (treated as
    old)."""
    if not published:
        return 0.0
    today = today or datetime.now(timezone.utc).date()
    try:
        released = datetime.fromisoformat(str(published)[:10]).date()
    except ValueError:
        return 0.0
    age_days = max(0, (today - released).days)
    return RECENCY_MAX_BOOST * 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)


def passes_threshold(match):
    """Adjusted score must clear the threshold, and the raw similarity may
    fall short of it by at most RECENCY_THRESHOLD_ALLOWANCE — so the recency
    bonus can tip a borderline match over the line but never carries a weak
    one in on its own."""
    t = threshold_for(match)
    return (match["adjusted_score"] >= t
            and match["score"] >= t - RECENCY_THRESHOLD_ALLOWANCE)


def rank_episodes(art, episodes, sims, use_recency=True, today=None):
    """Picks the TOP_MATCHES_PER_ARTICLE episodes for one article.

    1. Whether the article gets a match at all is decided by raw similarity,
       as before: its best raw match must be within
       RECENCY_THRESHOLD_ALLOWANCE of that episode's threshold.
    2. If so, the winner is the qualifying episode (raw similarity within
       the allowance of its own threshold) with the highest ADJUSTED score
       (similarity + recency bonus). Episodes that don't qualify never get a
       bonus, so a recent-but-weak episode can't win, and with no bonus
       this is exactly plain similarity ranking.
    """
    def floor_for(ep):
        t = FEED_SIMILARITY_THRESHOLD if ep.get("source") == "feed" else SIMILARITY_THRESHOLD
        return t - RECENCY_THRESHOLD_ALLOWANCE

    candidates = []
    for ep, sim in zip(episodes, sims):
        qualifies = sim >= floor_for(ep)
        boost = recency_boost(ep.get("published"), today) if use_recency and qualifies else 0.0
        candidates.append((qualifies, {
            "article": art,
            "episode": {"title": ep["title"], "url": ep["url"],
                        "source": ep.get("source", "transcript"),
                        "published": ep.get("published")},
            "score": round(float(sim), 3),
            "recency_boost": round(boost, 3),
            "adjusted_score": round(float(sim) + boost, 3),
        }))

    best_raw = max(candidates, key=lambda c: c[1]["score"])
    if best_raw[0]:
        pool = sorted((m for q, m in candidates if q),
                      key=lambda m: m["adjusted_score"], reverse=True)
    else:
        pool = sorted((m for _, m in candidates), key=lambda m: m["score"], reverse=True)
    return pool[:TOP_MATCHES_PER_ARTICLE]


def match_articles_to_episodes_tfidf(articles, episodes):
    """TF-IDF fallback: fits one vectorizer across episodes + articles
    together (required so they share a vocabulary/vector space), then scores
    cosine similarity. No caching — fitting TF-IDF over ~12 episodes and a
    day's worth of articles is fast enough to redo every run."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not episodes or not articles:
        return []

    ep_texts = [episode_text_for_matching(ep) for ep in episodes]
    art_texts = [f"{a['title']}. {a['summary']}" for a in articles]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2),
    )
    all_vectors = vectorizer.fit_transform(ep_texts + art_texts)
    ep_matrix = all_vectors[:len(ep_texts)]
    art_matrix = all_vectors[len(ep_texts):]

    matches = []
    for i, art in enumerate(articles):
        text = f"{art['title']}. {art['summary']}"
        if not text.strip():
            continue
        sims = cosine_similarity(art_matrix[i], ep_matrix)[0]
        # No recency bonus here: TF-IDF scores sit on a much lower scale
        # (threshold ~0.05), so a 0.04 bonus would swamp them.
        matches.extend(rank_episodes(art, episodes, sims, use_recency=False))

    matches.sort(key=lambda m: m["adjusted_score"], reverse=True)
    return matches


# ---------------------------------------------------------------------------
# AD HEADLINE DRAFTING
# ---------------------------------------------------------------------------
# Two backends, same as the embedding step: a free, offline, template-based
# generator (default — no API key, no cost, no network call) and an optional
# Claude-API-backed generator for more varied phrasing, used only if
# ANTHROPIC_API_KEY is set. Most days the templates below are indistinguishable
# in quality from what an LLM would produce for this exact "Continue learning"
# ad format, since the format itself is the whole idea — see AudienceLift.

FREE_HEADLINE_TEMPLATES = [
    "Continue learning: {title}",
    "Related listening for GPs: {title}",
    "More on this topic: {title}",
]


def _shorten_title(title, max_len=70):
    if len(title) <= max_len:
        return title
    return title[: max_len - 1].rsplit(" ", 1)[0] + "…"


def draft_headlines_free(matches):
    """Free, offline fallback: fixed 'Continue learning' style templates.
    No API call, no cost, works with no setup at all."""
    for m in matches:
        short_title = _shorten_title(m["episode"]["title"])
        m["headlines"] = [
            t.format(title=short_title) for t in FREE_HEADLINE_TEMPLATES[:HEADLINES_PER_MATCH]
        ]
    return matches


def draft_headlines(matches):
    """Adds draft native-ad headlines to the top matches. Uses the free
    template-based generator by default. If ANTHROPIC_API_KEY is set (and
    the `anthropic` package + a working API balance are available), uses
    Claude instead for more varied phrasing — this is optional, not
    required, and costs money per run. Falls back to the free templates on
    any failure (missing key, missing package, API error, low balance,
    etc.) so a run never comes back with no headlines at all."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set — using free template-based headlines "
              "(no cost, no API call). This is the default; see README if you "
              "ever want Claude-drafted headlines instead.")
        return draft_headlines_free(matches)

    try:
        import anthropic
    except ImportError:
        print("`anthropic` package not installed — using free template-based headlines instead.")
        return draft_headlines_free(matches)

    client = anthropic.Anthropic(api_key=api_key)

    for m in matches:
        prompt = f"""You write native-ad headlines for a medical podcast aimed at
GPs. A reader just read this article:

Title: {m['article']['title']}
Summary: {m['article']['summary']}

We want to advertise this podcast episode next to it:

Episode: {m['episode']['title']}

Write {HEADLINES_PER_MATCH} short native-ad headlines (under 12 words each)
in a "continue learning" style — informative, specific, not salesy. No
emojis, no clickbait, no exclamation marks. Return ONLY a JSON array of
strings, nothing else."""

        try:
            resp = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            m["headlines"] = json.loads(raw)
        except Exception as e:
            print(f"Headline drafting via Claude failed for '{m['episode']['title']}' "
                  f"({e}) — using a free template headline for this one instead.")
            short_title = _shorten_title(m["episode"]["title"])
            m["headlines"] = [
                t.format(title=short_title) for t in FREE_HEADLINE_TEMPLATES[:HEADLINES_PER_MATCH]
            ]

    return matches


# ---------------------------------------------------------------------------
# DASHBOARD GENERATION
# ---------------------------------------------------------------------------

DASHBOARD_TEMPLATE = "dashboard_template.html"
# Deliberately NOT "dashboard.html" — a file with that name was delivered to
# you as a read-only download earlier, and macOS (and this script) can never
# overwrite a read-only file no matter which computer it's on. Writing to a
# fresh filename that was always created locally by this script, never
# downloaded, sidesteps that permanently: this file is fully yours from the
# moment it's first created, so every future run can overwrite it normally.
DASHBOARD_OUTPUT = "dashboard_report.html"


def write_dashboard(output):
    """Generates dashboard_report.html from dashboard_template.html with
    today's matches baked directly into the page as JSON. This is
    deliberate: a page opened straight from disk (file://) is normally
    blocked by the browser from fetching other local files for security
    reasons, which is why relying on a dashboard fetching matches.json at
    load time was unreliable. Baking the data in at generation time
    sidesteps that completely — the generated file always shows exactly
    what was true when it was written, in every browser, no fetch involved."""
    if not os.path.exists(DASHBOARD_TEMPLATE):
        print(f"Note: {DASHBOARD_TEMPLATE} not found in this folder — skipping "
              f"dashboard generation (matches.json was still written normally, "
              f"you can load it manually via the 'Load matches.json' button in "
              f"any dashboard file you already have).")
        return

    with open(DASHBOARD_TEMPLATE, "r", encoding="utf-8") as f:
        template = f.read()

    # Escape "</" so a literal "</script>" can never appear inside the
    # embedded JSON and prematurely close the script tag.
    json_str = json.dumps(output).replace("</", "<\\/")
    html = template.replace("__MATCHES_JSON__", json_str)

    try:
        with open(DASHBOARD_OUTPUT, "w", encoding="utf-8") as f:
            f.write(html)
    except PermissionError:
        print(f"Couldn't write {DASHBOARD_OUTPUT} — it exists but isn't writable "
              f"(probably marked read-only). matches.json was still written "
              f"normally. Fix: delete or rename the existing {DASHBOARD_OUTPUT} "
              f"and run this again, or check its permissions (right-click in "
              f"Finder -> Get Info -> uncheck 'Locked', or Terminal: "
              f"chmod +w \"{DASHBOARD_OUTPUT}\").")
        return
    print(f"Wrote {DASHBOARD_OUTPUT} (today's matches baked in — just open it, no loading step needed)")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    backend = get_embedding_backend()

    print("Loading episodes...")
    episodes, episode_source = load_episodes_for_matching(backend)
    print(f"  {len(episodes)} episodes loaded (source: {episode_source})")

    print("Fetching recent articles...")
    articles = fetch_recent_articles()
    print(f"  {len(articles)} articles found via live RSS in the last {LOOKBACK_HOURS}h")

    used_offline_cache = False
    if not articles:
        articles = load_offline_articles_cache()
        used_offline_cache = bool(articles)

    print("Matching...")
    if backend == "sentence-transformers":
        embedder = load_embedder()
        if episode_source == "local":
            print("Building/loading episode embeddings...")
            episodes = build_or_load_episode_embeddings(episodes, embedder)
            episodes = add_feed_episodes_from_index(episodes)
        # else: episode_source == "index" — embeddings already computed,
        # loaded straight from EPISODE_INDEX_FILE, nothing to build.
        matches_all = match_articles_to_episodes_st(articles, episodes, embedder)
    else:
        matches_all = match_articles_to_episodes_tfidf(articles, episodes)

    print(f"\n  Backend: {backend}")
    print(f"  Score table (top match per article, before thresholding):")
    print(f"  ([f] = episode embedded from feed show notes, threshold "
          f"{FEED_SIMILARITY_THRESHOLD}; others from transcripts, threshold "
          f"{SIMILARITY_THRESHOLD})")
    print(f"  (similarity + recency bonus = adjusted; released date in brackets)")
    for m in matches_all:
        tag = "[f]" if m["episode"].get("source") == "feed" else "   "
        keep = "*" if passes_threshold(m) else " "
        print(f"    {m['score']:.3f}+{m['recency_boost']:.3f}={m['adjusted_score']:.3f}{keep} "
              f"{m['article']['title'][:55]!r:58s} -> {tag} {m['episode']['title'][:45]} "
              f"[{m['episode'].get('published') or 'undated'}]")

    matches = [m for m in matches_all if passes_threshold(m)]
    print(f"\n  {len(matches)} matches above threshold (* above; "
          f"{SIMILARITY_THRESHOLD} transcript / {FEED_SIMILARITY_THRESHOLD} show notes) "
          f"of {len(matches_all)} article-episode pairs scored")

    print("Drafting ad headlines for top matches...")
    matches = draft_headlines(matches)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lookback_hours": LOOKBACK_HOURS,
        "threshold": SIMILARITY_THRESHOLD,
        "feed_threshold": FEED_SIMILARITY_THRESHOLD,
        "recency": {
            "max_boost": RECENCY_MAX_BOOST,
            "half_life_days": RECENCY_HALF_LIFE_DAYS,
            "threshold_allowance": RECENCY_THRESHOLD_ALLOWANCE,
        },
        "embedding_backend": backend,
        "used_offline_article_cache": used_offline_cache,
        "match_count": len(matches),
        "matches": matches,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {OUTPUT_FILE}")

    write_dashboard(output)


if __name__ == "__main__":
    main()
