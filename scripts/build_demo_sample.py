"""OFFLINE — build the curated ~150-candidate demo pool + its tiny artifacts.

The Caliber demo (HuggingFace Spaces / Gradio, built in a follow-up) runs the
REAL ``src/caliber`` ranking pipeline live, but on a small curated pool instead
of the full 100K so a judge sees it work in seconds. This script builds that
pool. It does NOT re-implement any ranking, detection, or encoding logic — it
selects a real *subset* of ``data/candidates.jsonl`` (no synthetic records) and
reuses the exact canonical code the pipeline uses:

    - honeypot detection      -> caliber.honeypot.is_honeypot   (the floor rule)
    - keyword-stuffer gating  -> eval.heuristics.stuffer_reasons (the gate rule)
    - career-substance reading-> eval.heuristics (title_class, substance areas)
    - encoding + FAISS index  -> caliber.embeddings / caliber.index (precompute's
                                 own building blocks, not a parallel encoder)

The pool is curated to TELL THE "substance over keywords" STORY when ranked:

    genuine strong fits  -> rank top (real retrieval/ranking/ML substance, India,
                            5-9yr band) — what the JD actually wants.
    keyword-stuffers     -> sink (non-tech title + many AI skill TAGS, zero
                            corroborating career substance) — the skill-gate buries them.
    honeypots            -> floored (internally impossible profiles) — the
                            consistency detector forces them down + flags them.
    hidden gems          -> surface despite plain titles (adjacent title, real
                            retrieval/ranking work in the descriptions) — the
                            "plain-language Tier-5" the JD explicitly wants found.
    noise filler         -> realistic background so the pool feels like the real pool.

Everything is deterministic (``config.SEED``); two runs produce byte-identical
output. Selection is by file-order + stable sorts + a seeded RNG only for the
noise sample.

Usage:
    CALIBER_ALLOW_MODEL_DOWNLOAD=1 python scripts/build_demo_sample.py
    # outputs under sandbox/demo_data/ (demo_candidates.jsonl + artifacts/)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Make ``src/`` (caliber) and the repo-root ``eval/`` package-less modules
# importable when run as a plain script, exactly like scripts/precompute.py.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "eval"))

from caliber import config  # noqa: E402
from caliber.embeddings import encode_candidates  # noqa: E402
from caliber.honeypot import is_honeypot  # noqa: E402
from caliber.index import build_index, save_index  # noqa: E402

# eval/heuristics.py is the canonical stuffer detector + the shared
# title/substance lexicons (it itself delegates honeypot detection to
# caliber.honeypot, so both halves agree). Imported, never copied.
import heuristics  # noqa: E402  (resolved via sys.path -> repo_root/eval)


# --------------------------------------------------------------------------- #
# Curation targets. Small, deliberate counts — the demo only needs enough of
# each category to make the story legible, not a representative sample.
# --------------------------------------------------------------------------- #
TARGET_TOTAL = 150
N_GENUINE = 10      # strong fits that should rank top
N_STUFFER = 4       # keyword-stuffers the skill-gate should bury
N_HONEYPOT = 4      # impossible profiles the consistency detector should floor
N_HIDDEN_GEM = 12   # plain-title candidates with real substance (Tier-5s)
# remainder -> realistic noise filler, sampled deterministically.

# IR-aligned substance areas (NLP/IR/retrieval/ranking) vs the generic
# applied-ml-in-prod area. A genuine Senior-AI-Engineer fit must show at least
# one of these, not merely "deployed a model" (STRATEGY §4.1 / §4.3).
_IR_AREAS = {"retrieval_embeddings", "ranking_ltr", "recommendation", "search_ir"}

DEFAULT_OUT_DIR = _REPO_ROOT / "sandbox" / "demo_data"


def stream_records(path: Path) -> "Any":
    """Yield candidate records from a JSONL file, one decoded line at a time."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _country(rec: Dict[str, Any]) -> str:
    return (rec.get("profile", {}).get("country") or "").strip().lower()


def _yoe(rec: Dict[str, Any]) -> float:
    try:
        return float(rec.get("profile", {}).get("years_of_experience") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _title(rec: Dict[str, Any]) -> str:
    return (rec.get("profile", {}).get("current_title") or "").strip()


# --------------------------------------------------------------------------- #
# Per-candidate lightweight metadata gathered in ONE streaming pass. We keep
# only small scalars (no full records) so classifying the 100K pool stays
# memory-cheap; the selected full records are re-read in a second pass.
# --------------------------------------------------------------------------- #
class Meta:
    __slots__ = ("cid", "category", "n_substance", "n_ir", "n_ai_skills",
                 "hp_reasons", "title")

    def __init__(self, cid: str, category: str, n_substance: int, n_ir: int,
                 n_ai_skills: int, hp_reasons: List[str], title: str) -> None:
        self.cid = cid
        self.category = category
        self.n_substance = n_substance
        self.n_ir = n_ir
        self.n_ai_skills = n_ai_skills
        self.hp_reasons = hp_reasons
        self.title = title


def classify(rec: Dict[str, Any]) -> Meta:
    """Assign one candidate to a curation category, reusing the canonical
    detectors. Priority order matters: honeypot and stuffer are checked FIRST so
    an impossible/keyword-stuffed profile is never mistaken for a genuine fit.
    """
    cid = rec["candidate_id"]
    title = _title(rec)
    text = heuristics.career_text(rec)
    areas = heuristics.substance_areas_hit(text)
    n_substance = len(areas)
    n_ir = len(set(areas) & _IR_AREAS)
    n_ai_skills = heuristics.ai_skill_count(rec)
    tc = heuristics.title_class(title)
    yoe = _yoe(rec)
    india = _country(rec) == "india"

    hp_flag, hp_reasons = is_honeypot(rec)
    if hp_flag:
        cat = "honeypot"
    elif heuristics.stuffer_reasons(rec):
        cat = "stuffer"
    elif india and tc == "strong" and 5 <= yoe <= 9 and n_substance >= 2 and n_ir >= 1:
        cat = "genuine"
    elif india and tc == "adjacent" and 4 <= yoe <= 10 and n_substance >= 1:
        # n_substance >= 1 (not 2): adjacent-titled candidates with real
        # retrieval/ranking/ML work span many titles (Data/Backend/Analytics
        # Engineer, ...) at this threshold; at >= 2 the pool collapses to a
        # single title. The looser gate keeps the "plain-language Tier-5" story
        # while letting the selection diversify across titles for the demo.
        cat = "hidden_gem"
    else:
        cat = "noise"

    return Meta(cid, cat, n_substance, n_ir, n_ai_skills, hp_reasons, title)


def _round_robin(buckets: Dict[str, List[Meta]], ordered_keys: List[str],
                 k: int) -> List[Meta]:
    """Pick ``k`` items spread across ``buckets`` for variety: take the best from
    each bucket in ``ordered_keys`` order, then the second-best, and so on. Each
    bucket must already be internally sorted best-first; ``ordered_keys`` fixes a
    deterministic visiting order. Used to diversify honeypots (across failure
    modes) and hidden gems (across job titles)."""
    picked: List[Meta] = []
    while len(picked) < k and any(buckets[key] for key in ordered_keys):
        for key in ordered_keys:
            if buckets[key]:
                picked.append(buckets[key].pop(0))
                if len(picked) == k:
                    return picked
    return picked


def select(metas: List[Meta]) -> Tuple[Dict[str, List[Meta]], List[str]]:
    """Apply the curation targets to the classified pool. Returns the chosen
    Meta per category and the full ordered list of selected candidate_ids.
    All sorts are stable with candidate_id as the final tie-break (determinism).
    """
    by_cat: Dict[str, List[Meta]] = {}
    for m in metas:
        by_cat.setdefault(m.category, []).append(m)

    chosen: Dict[str, List[Meta]] = {}

    # Genuine: most real IR/ranking substance first, then breadth of substance.
    genuine = sorted(by_cat.get("genuine", []),
                     key=lambda m: (-m.n_ir, -m.n_substance, m.cid))
    chosen["genuine"] = genuine[:N_GENUINE]

    # Stuffers: most egregious first (most AI skill tags with zero substance).
    stuffers = sorted(by_cat.get("stuffer", []),
                      key=lambda m: (-m.n_ai_skills, m.cid))
    chosen["stuffer"] = stuffers[:N_STUFFER]

    # Honeypots: spread across distinct contradiction types (reason-first-word
    # buckets, each pre-sorted by candidate_id), so the demo shows the detector's
    # range rather than four copies of one trip.
    hp_buckets: Dict[str, List[Meta]] = {}
    for m in sorted(by_cat.get("honeypot", []), key=lambda m: m.cid):
        key = m.hp_reasons[0].split()[0] if m.hp_reasons else "other"
        hp_buckets.setdefault(key, []).append(m)
    chosen["honeypot"] = _round_robin(hp_buckets, sorted(hp_buckets), N_HONEYPOT)

    # Hidden gems: plain (adjacent) titles with real substance. Diversify across
    # job TITLES so the demo shows the "Tier-5 hides under many titles" story
    # (Data/Backend/Analytics Engineer, ...) instead of 12 of one title. Each
    # title bucket is sorted strongest-substance-first; titles are visited
    # strongest-example-first then alphabetically (deterministic).
    gem_buckets: Dict[str, List[Meta]] = {}
    for m in by_cat.get("hidden_gem", []):
        gem_buckets.setdefault(m.title, []).append(m)
    for title in gem_buckets:
        gem_buckets[title].sort(key=lambda m: (-m.n_substance, -m.n_ir, m.cid))
    title_order = sorted(gem_buckets,
                         key=lambda t: (-gem_buckets[t][0].n_substance, t))
    chosen["hidden_gem"] = _round_robin(gem_buckets, title_order, N_HIDDEN_GEM)

    selected_ids = {m.cid for ms in chosen.values() for m in ms}

    # Noise filler: deterministic seeded sample of everything not already chosen
    # (and not itself a stuffer/honeypot we just didn't pick — keep the labeled
    # categories clean so the summary's planted ids are the only ones of each).
    n_needed = TARGET_TOTAL - len(selected_ids)
    noise_pool = sorted(
        (m for m in metas
         if m.cid not in selected_ids
         and m.category in ("noise", "hidden_gem", "genuine")),
        key=lambda m: m.cid,
    )
    rng = random.Random(config.SEED)
    noise = rng.sample(noise_pool, min(n_needed, len(noise_pool))) if noise_pool else []
    chosen["noise"] = sorted(noise, key=lambda m: m.cid)

    all_ids = sorted({m.cid for ms in chosen.values() for m in ms})
    return chosen, all_ids


def build_artifacts(records: List[Dict[str, Any]], artifacts_dir: Path) -> Tuple[int, int]:
    """Encode the demo records and build the FAISS index the SAME way precompute
    does (caliber.embeddings + caliber.index), into ``artifacts_dir``. Tiny pool,
    so this runs in seconds on CPU. Returns (n_rows, dim).
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ids = [r["candidate_id"] for r in records]

    # Offline prep is allowed to download the model (cached under models/ already).
    os.environ.setdefault("CALIBER_ALLOW_MODEL_DOWNLOAD", "1")

    emb = encode_candidates(records)
    if emb.shape[0] != len(ids):
        raise RuntimeError(f"row/id mismatch: {emb.shape[0]} emb vs {len(ids)} ids")

    np.save(artifacts_dir / config.CANDIDATE_IDS_FILE, np.array(ids))
    np.save(artifacts_dir / config.CANDIDATE_EMB_FILE, emb)
    save_index(build_index(emb), artifacts_dir / config.FAISS_INDEX_FILE)

    # The demo's artifacts dir must be a COMPLETE rank-ready set: rank.py's
    # load_artifacts also wants jd_profile.json (the same JD profile as the full
    # run — it does not depend on the pool). Copy it in if available so the Space
    # can point rank.py straight at this directory.
    src_jd = config.ARTIFACTS_DIR / config.JD_PROFILE_FILE
    if src_jd.exists():
        shutil.copyfile(src_jd, artifacts_dir / config.JD_PROFILE_FILE)

    return emb.shape[0], emb.shape[1]


def print_summary(chosen: Dict[str, List[Meta]], total: int, pool_size: int,
                  detected_stuffers: int, detected_honeypots: int) -> None:
    print("\n" + "=" * 72)
    print(f"CALIBER DEMO SAMPLE — {total} candidates curated from {pool_size:,} pool")
    print("=" * 72)
    labels = [
        ("genuine",    "genuine strong fits (should rank TOP)"),
        ("hidden_gem", "hidden gems — plain title, real substance (should surface)"),
        ("stuffer",    "keyword-stuffers (skill-gate should BURY)"),
        ("honeypot",   "honeypots — impossible profiles (detector should FLOOR)"),
        ("noise",      "realistic noise filler"),
    ]
    for key, desc in labels:
        print(f"  {len(chosen.get(key, [])):>3}  {desc}")
    print("-" * 72)
    print(f"  whole pool contained {detected_stuffers:,} stuffers and "
          f"{detected_honeypots:,} honeypots in total")

    print("\nPlanted KEYWORD-STUFFERS (verify the demo buries these):")
    for m in chosen.get("stuffer", []):
        print(f"  {m.cid}  {m.n_ai_skills:>2} AI skills, 0 substance  | {m.title!r}")

    print("\nPlanted HONEYPOTS (verify the demo floors + flags these):")
    for m in chosen.get("honeypot", []):
        reason = m.hp_reasons[0] if m.hp_reasons else "?"
        print(f"  {m.cid}  {m.title!r}")
        print(f"            -> {reason}")

    print("\nGenuine strong fits (should rank top):")
    for m in chosen.get("genuine", []):
        print(f"  {m.cid}  IR-areas={m.n_ir} substance={m.n_substance}  | {m.title!r}")

    print("\nHidden gems (plain title, real substance — should surface):")
    for m in chosen.get("hidden_gem", []):
        print(f"  {m.cid}  substance={m.n_substance}  | {m.title!r}")
    print("=" * 72 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=config.CANDIDATES_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR,
                        help="demo output dir (jsonl + artifacts/ written here)")
    args = parser.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = out_dir / "artifacts"

    # Pass 1 — classify the whole pool (small scalars only; memory-cheap).
    print(f"[demo] pass 1: classifying pool from {args.candidates} ...")
    t0 = time.time()
    metas: List[Meta] = []
    detected_stuffers = detected_honeypots = 0
    for rec in stream_records(args.candidates):
        m = classify(rec)
        if m.category == "stuffer":
            detected_stuffers += 1
        elif m.category == "honeypot":
            detected_honeypots += 1
        metas.append(m)
    pool_size = len(metas)
    print(f"[demo]   classified {pool_size:,} candidates in {time.time() - t0:.1f}s")

    chosen, selected_ids = select(metas)
    selected_set = set(selected_ids)
    cat_of = {m.cid: key for key, ms in chosen.items() for m in ms}

    # Pass 2 — re-read only the selected records (full dicts) in file order.
    print(f"[demo] pass 2: extracting {len(selected_set)} selected records ...")
    picked: Dict[str, Dict[str, Any]] = {}
    for rec in stream_records(args.candidates):
        cid = rec["candidate_id"]
        if cid in selected_set:
            picked[cid] = rec

    missing = selected_set - set(picked)
    if missing:
        raise RuntimeError(f"selected ids not found on second pass: {sorted(missing)[:5]} ...")

    # Stable file order: candidate_id ascending (the canonical tie-break key).
    records = [picked[cid] for cid in selected_ids]

    jsonl_path = out_dir / "demo_candidates.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[demo] wrote {len(records)} records -> {jsonl_path}")

    # Build the tiny artifacts the same way precompute does.
    print(f"[demo] encoding {len(records)} candidates + building FAISS index ...")
    t0 = time.time()
    n_rows, dim = build_artifacts(records, artifacts_dir)
    print(f"[demo]   built artifacts ({n_rows} rows x {dim} dims) in {time.time() - t0:.1f}s "
          f"-> {artifacts_dir}")

    print_summary(chosen, len(records), pool_size, detected_stuffers, detected_honeypots)


if __name__ == "__main__":
    main()
