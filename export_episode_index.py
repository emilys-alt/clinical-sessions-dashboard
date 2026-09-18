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


def main():
    print("Loading episodes from ./episodes/ ...")
    episodes = me.load_episodes()
    print(f"  {len(episodes)} episodes found")

    if not episodes:
        print("No episodes found in ./episodes/ — nothing to export.")
        sys.exit(1)

    print(f"Loading embedding model ({me.EMBED_MODEL_NAME})...")
    embedder = SentenceTransformer(me.EMBED_MODEL_NAME)

    print("Computing embeddings (transcript text is used here only, never written out)...")
    index_episodes = []
    for ep in episodes:
        text_for_embedding = me.episode_text_for_matching(ep)
        embedding = embedder.encode(text_for_embedding)
        index_episodes.append({
            "title": ep["title"],
            "url": ep["url"],
            "embedding": [round(float(x), 6) for x in embedding],
        })

    out = {
        "model": me.EMBED_MODEL_NAME,
        "episode_count": len(index_episodes),
        "episodes": index_episodes,
    }

    with open(me.EPISODE_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\nWrote {me.EPISODE_INDEX_FILE} — {len(index_episodes)} episodes, "
          f"titles + URLs + embedding vectors only, no transcript text.")
    print("Safe to commit and push this file to your public GitHub repo.")
    print("Do NOT commit the episodes/ folder itself — check .gitignore excludes it.")


if __name__ == "__main__":
    main()
