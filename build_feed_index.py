"""
Build Episode Index from the public podcast RSS feeds
=====================================================

Adds every episode from ArmchairMedical's public Captivate feeds to
episode_index.json, so matching_engine.py can match news articles against
the whole catalogue instead of only the episodes with local transcripts.

Each feed episode is embedded from its TITLE + SHOW-NOTES DESCRIPTION (the
public text already in the RSS feed), using the same sentence-transformers
model as the rest of the index. No transcripts are needed, so this runs fine
on a GitHub Actions runner — the daily workflow runs it before matching, so
new episodes are picked up automatically.

HOW IT MERGES WITH THE TRANSCRIPT-BASED INDEX
- Entries written by export_episode_index.py (source "transcript") are kept
  exactly as they are — transcript embeddings are higher quality, so they win.
- A feed episode that is the same episode as a transcript entry (same URL, or
  near-identical title) is skipped.
- The General Practice and Primary Care Masterclass feeds cross-post each
  other's episodes as "bonus" items (same link, title suffixed with the other
  show's name). Those are collapsed to one entry.
- Embeddings are reused when an episode's title + description haven't
  changed, so a daily run only encodes new/edited episodes and the committed
  file doesn't churn.
- If a feed can't be fetched, that feed's previous entries are kept rather
  than dropped, so one flaky feed never shrinks the index.

USAGE
    python build_feed_index.py
"""

import difflib
import hashlib
import html
import json
import re
import socket
import sys
import urllib.error
import urllib.request

import feedparser

EPISODE_INDEX_FILE = "episode_index.json"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # must match matching_engine.py
FEED_TIMEOUT_SECONDS = 20

CAPTIVATE_FEEDS = [
    "https://feeds.captivate.fm/general-practice/",
    "https://feeds.captivate.fm/armchairmedical/",
    "https://feeds.captivate.fm/womens-health-essentials/",
    "https://feeds.captivate.fm/health-across-the-ages/",
    "https://feeds.captivate.fm/mental-health-sessions/",
]

# Words of show-notes text to embed after the title. all-MiniLM-L6-v2 only
# reads the first 256 word-pieces anyway; the episode summary is always at
# the top of the notes.
DESCRIPTION_WORDS = 180

# Two titles at or above this similarity (after normalising) are treated as
# the same episode — but only if they also open with the same words, because
# series like "Hip / Knee / Shoulder Osteoarthritis and Surgical
# Interventions" differ by a single leading word and score ~0.9 regardless.
TITLE_DUPLICATE_RATIO = 0.9
_STOPWORDS = {"the", "a", "an", "of", "and", "in", "on", "for", "to", "with"}

# The show notes end with the same marketing block on every episode
# ("Enjoyed Today's Clinical Sessions Podcast? ... earn CPD hours ...").
# Left in, it pulls every episode's embedding towards the same point and
# blurs the matching, so cut the notes at the first of these markers.
BOILERPLATE_MARKERS = [
    r"Enjoyed Today'?s",
    r"If you are a General Practitioner",
    r"-{10,}",
    r"Learn more\s*https?://(www\.)?armchairmedical",
    r"Earn Educational Activity",
    r"Listen to your Clinical Sessions Podcasts",
    r"Continuing Professional Development \(CPD\)",
    r"Australian GPs can (also )?claim CPD",
    r"GPs can (also )?earn CPD",
]
_BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_MARKERS), re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def html_to_text(raw):
    text = re.sub(r"</(p|h\d|li|div)>|<br\s*/?>", "\n", raw or "", flags=re.I)
    text = html.unescape(_TAG_RE.sub(" ", text))
    return text


def clean_description(raw):
    text = html_to_text(raw)
    m = _BOILERPLATE_RE.search(text)
    if m:
        text = text[:m.start()]
    return _WS_RE.sub(" ", text).strip()


def strip_show_suffix(title, show_titles):
    """'Episode X - General Practice Clinical Sessions Podcast' -> 'Episode X'
    (cross-posted bonus items carry the original show's name)."""
    t = title.strip()
    for show in show_titles:
        for sep in (" - ", " – ", " | "):
            suffix = sep + show
            if t.lower().endswith(suffix.lower()):
                return t[: -len(suffix)].strip()
    return t


def normalise_title(title):
    t = html.unescape(title).lower()
    t = re.sub(r"\s[-–|]\s(primary care masterclass|general practice clinical sessions)[^-–|]*$", "", t)
    t = re.sub(r"\b(dr|prof|professor|assoc|aprof|a/prof|mr|ms|mrs|with)\b\.?", " ", t)
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return " ".join(w for w in t.split() if w not in _STOPWORDS)


def same_title(a, b):
    """True if two titles name the same episode: identical after
    normalising, one is the other plus extra words (e.g. a speaker name
    appended), or near-identical with the same opening words."""
    na, nb = normalise_title(a), normalise_title(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    ta, tb = na.split(), nb.split()
    small, big = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if len(small) >= 4 and big[:len(small)] == small:
        return True
    return (ta[:2] == tb[:2]
            and difflib.SequenceMatcher(None, na, nb).ratio() >= TITLE_DUPLICATE_RATIO)


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Feed fetching
# ---------------------------------------------------------------------------

def fetch_feed(url):
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (compatible; PodcastMatcher/1.0)"}
    )
    with urllib.request.urlopen(req, timeout=FEED_TIMEOUT_SECONDS) as resp:
        return feedparser.parse(resp.read())


def episode_url(entry):
    """Prefer the episode's own page. Some shows' feeds only link to the show
    homepage, so fall back to Captivate's public player page for that
    episode (keyed by the item GUID), which always resolves to it."""
    link = (entry.get("link") or "").strip()
    if "/episode/" in link:
        return link
    guid = (entry.get("id") or "").strip()
    if re.fullmatch(r"[0-9a-fA-F-]{36}", guid):
        return f"https://player.captivate.fm/episode/{guid}"
    return link


def collect_feed_episodes():
    """Returns (episodes, failed_feed_urls). Each episode is a dict with
    title, url, feed, text (what gets embedded) and is_full."""
    parsed = []
    failed = []
    for url in CAPTIVATE_FEEDS:
        print(f"  Fetching {url} ...", end=" ", flush=True)
        try:
            feed = fetch_feed(url)
        except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
            print(f"unreachable — keeping its previous entries ({e})")
            failed.append(url)
            continue
        except Exception as e:  # noqa: BLE001
            print(f"failed — keeping its previous entries ({e})")
            failed.append(url)
            continue
        if not feed.entries:
            print("no entries — keeping its previous entries")
            failed.append(url)
            continue
        print(f"{len(feed.entries)} episodes ({feed.feed.get('title', '?')})")
        parsed.append((url, feed))

    show_titles = [f.feed.get("title", "") for _, f in parsed if f.feed.get("title")]
    show_titles += ["General Practice Clinical Sessions Podcast", "Primary Care Masterclass Podcast"]

    episodes = []
    for url, feed in parsed:
        for e in feed.entries:
            title = strip_show_suffix(html.unescape(e.get("title", "")), show_titles)
            if not title:
                continue
            desc = clean_description(e.get("summary") or e.get("description", ""))
            words = desc.split()
            # The notes usually open by repeating the title — drop it so the
            # title isn't double-weighted.
            if desc.lower().startswith(title.lower()):
                words = desc[len(title):].split()
            text = title + ". " + " ".join(words[:DESCRIPTION_WORDS])
            episodes.append({
                "title": title,
                "url": episode_url(e),
                "feed": url,
                "text": text,
                "is_full": e.get("itunes_episodetype", "full") != "bonus",
            })
    return episodes, failed


# ---------------------------------------------------------------------------
# Merge + embed
# ---------------------------------------------------------------------------

def load_index():
    try:
        with open(EPISODE_INDEX_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}
    eps = data.get("episodes", [])
    for ep in eps:
        # Entries from before this script existed all came from transcripts.
        ep.setdefault("source", "transcript")
    return eps


def write_index(out):
    """Pretty JSON, but each embedding vector on a single line, so a git
    diff shows one changed line per added/edited episode instead of 384."""
    text = json.dumps(out, indent=2)
    text = re.sub(
        r'"embedding": \[\s*([^\]]*?)\s*\]',
        lambda m: '"embedding": [' + ", ".join(x.strip() for x in m.group(1).split(",")) + "]",
        text,
    )
    with open(EPISODE_INDEX_FILE, "w", encoding="utf-8") as f:
        f.write(text + "\n")


def main():
    existing = load_index()
    transcript_eps = [e for e in existing if e["source"] == "transcript"]
    old_feed_eps = [e for e in existing if e["source"] == "feed"]
    old_by_hash = {e.get("text_hash"): e for e in old_feed_eps if e.get("text_hash")}

    print("Fetching podcast feeds...")
    feed_eps, failed_feeds = collect_feed_episodes()

    # Original ("full") items before cross-posted "bonus" copies, so the copy
    # that survives de-duplication is the one from the episode's home show.
    feed_eps.sort(key=lambda e: not e["is_full"])

    kept = []
    skipped_transcript = []
    skipped_dupe = 0
    for ep in feed_eps:
        twin = next((t for t in transcript_eps
                     if t["url"] == ep["url"] or same_title(t["title"], ep["title"])), None)
        if twin:
            skipped_transcript.append((ep["title"], twin["title"]))
            continue
        if any(k["url"] == ep["url"] or same_title(k["title"], ep["title"]) for k in kept):
            skipped_dupe += 1
            continue
        kept.append(ep)

    # Keep last run's entries for any feed we couldn't fetch this time.
    for old in old_feed_eps:
        if old.get("feed") in failed_feeds and not any(
            k["url"] == old["url"] or same_title(k["title"], old["title"]) for k in kept
        ):
            kept.append({**old, "_reuse": True})

    to_embed = []
    for ep in kept:
        if ep.get("_reuse"):
            continue
        ep["text_hash"] = text_hash(ep["text"])
        prev = old_by_hash.get(ep["text_hash"])
        if prev:
            ep["embedding"] = prev["embedding"]
        else:
            to_embed.append(ep)

    if to_embed:
        print(f"Embedding {len(to_embed)} new or changed episodes with {EMBED_MODEL_NAME}...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBED_MODEL_NAME)
        vectors = model.encode([ep["text"] for ep in to_embed], batch_size=32)
        for ep, vec in zip(to_embed, vectors):
            ep["embedding"] = [round(float(x), 6) for x in vec]
    else:
        print("No new or changed episodes — reusing all existing embeddings.")

    new_feed_entries = []
    for ep in kept:
        new_feed_entries.append({
            "title": ep["title"],
            "url": ep["url"],
            "source": "feed",
            "feed": ep["feed"],
            "text_hash": ep["text_hash"],
            "embedding": ep["embedding"],
        })

    episodes = transcript_eps + new_feed_entries
    out = {
        "model": EMBED_MODEL_NAME,
        "episode_count": len(episodes),
        "transcript_episode_count": len(transcript_eps),
        "feed_episode_count": len(new_feed_entries),
        "episodes": episodes,
    }

    if not new_feed_entries and not transcript_eps:
        print("ERROR: no episodes at all — not overwriting the index.")
        sys.exit(1)

    write_index(out)

    print(f"\nWrote {EPISODE_INDEX_FILE}: {len(episodes)} episodes "
          f"({len(transcript_eps)} transcript-based, {len(new_feed_entries)} from feeds; "
          f"{len(to_embed)} newly embedded).")
    print(f"  Skipped {len(skipped_transcript)} feed episodes already covered by transcripts, "
          f"{skipped_dupe} cross-posted/duplicate feed items.")
    if failed_feeds:
        print(f"  WARNING: {len(failed_feeds)} feed(s) failed this run; kept their previous entries: "
              + ", ".join(failed_feeds))


if __name__ == "__main__":
    main()
