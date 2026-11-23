"""
Export Episode Index (for hosting the dashboard on the web)
=============================================================

Run this LOCALLY, on your own machine, whenever you add or change episodes.
It reads your full transcripts from ./episodes/ and writes episode_index.json
containing ONLY:
    - episode title
    - episode url
    - a pre-computed embedding vector (a list of ~384 numbers — not text,
      not reversible back into anything readable)

It does NOT include any transcript text. That's the whole point: this file
is safe to commit to a public GitHub repo, because there's nothing in it a
reader could use to reconstruct or read the actual episode content — just
titles and links you'd share publicly anyway, plus numbers a matching
algorithm uses.

WHY THIS EXISTS
Free GitHub Pages requires a public repository. Your episode transcripts are
the same premium content ArmchairMedical.tv charges for — you don't want the
full text of those sitting in a public repo anyone can browse. This script
lets the cloud-hosted dashboard do its matching without ever uploading the
transcripts themselves.

USAGE
    python export_episode_index.py

The index also holds every other episode from the public podcast feeds,
embedded from title + show notes by build_feed_index.py (the daily GitHub
Action refreshes those). This script only replaces the transcript-based
entries and keeps the feed-based ones, dropping any feed entry that now has
a transcript.

Then commit and push the updated episode_index.json to your GitHub repo
(but never commit the episodes/ folder itself — see .gitignore).

Requires sentence-transformers to already be installed (same as
matching_engine.py) — run this after your normal setup, in the same venv.
"""

import json
import sys

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("sentence-transformers isn't installed. Run this inside the same "
          "environment matching_engine.py uses (e.g. activate venv/ first), "
          "or run:\n    pip install sentence-transformers")
    sys.exit(1)

import matching_engine as me
import build_feed_index as bfi


def main():
    print("Loading episodes from ./episodes/ ...")
    episodes = me.load_episodes()
    print(f"  {len(episodes)} episodes found")

    if not episodes:
        print("No episodes found in ./episodes/ — nothing to export.")
        sys.exit(1)

    print(f"Loading embedding model ({me.EMBED_MODEL_NAME})...")
    embedder = SentenceTransformer(me.EMBED_MODEL_NAME)

    existing = bfi.load_index()

    print("Computing embeddings (transcript text is used here only, never written out)...")
    index_episodes = []
    for ep in episodes:
        text_for_embedding = me.episode_text_for_matching(ep)
        embedding = embedder.encode(text_for_embedding)
        index_episodes.append({
            "title": ep["title"],
            "url": ep["url"],
            # Release date from the existing index (filled in from the feeds
            # by build_feed_index.py); None until the next feed refresh.
            "published": next(
                (e.get("published") for e in existing
                 if e["url"] == ep["url"] or bfi.same_title(e["title"], ep["title"])),
                None,
            ),
            "source": "transcript",
            "embedding": [round(float(x), 6) for x in embedding],
        })

    # Keep the feed-based entries added by build_feed_index.py, minus any
    # episode that now has a (better) transcript-based embedding.
    feed_episodes = [
        e for e in existing
        if e["source"] == "feed" and not any(
            t["url"] == e["url"] or bfi.same_title(t["title"], e["title"])
            for t in index_episodes
        )
    ]
    episodes = index_episodes + feed_episodes

    out = {
        "model": me.EMBED_MODEL_NAME,
        "episode_count": len(episodes),
        "transcript_episode_count": len(index_episodes),
        "feed_episode_count": len(feed_episodes),
        "episodes": episodes,
    }

    bfi.write_index(out)

    print(f"\nWrote {me.EPISODE_INDEX_FILE} — {len(index_episodes)} transcript-based "
          f"episodes (titles + URLs + embedding vectors only, no transcript text), "
          f"plus {len(feed_episodes)} feed-based episodes kept from build_feed_index.py.")
    print("Safe to commit and push this file to your public GitHub repo.")
    print("Do NOT commit the episodes/ folder itself — check .gitignore excludes it.")


if __name__ == "__main__":
    main()
