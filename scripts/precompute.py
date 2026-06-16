"""OFFLINE precompute — build all artifacts ``rank.py`` depends on.

No time limit; runs once on the full 100K pool and persists to ``artifacts/``:

    1. parse the JD -> ``jd_profile.json`` (jd_profile)            [other owner]
    2. build rich per-candidate text + encode -> ``candidate_emb.npy``
       (embeddings)                                                [this module]
    3. build the FAISS index -> ``faiss.index`` (+ id mapping)     [this module]
    4. fit/persist BM25 state over role descriptions              [other owner]
    5. record precompute wall-clock into ``submission_metadata.yaml``

This is allowed to be slow and may use an LLM for offline label work elsewhere,
but everything it emits is static so the online ranker stays CPU-only and
network-free.

Currently wires steps 2 and 3 (the embedding + index half, feat/embeddings).
Steps 1 and 4 are stubbed as TODOs owned by the other half of the team.

Usage:
    python scripts/precompute.py [--candidates data/candidates.jsonl]
                                 [--out artifacts] [--batch-size 256] [--limit N]
    python scripts/precompute.py --sanity   # quick ML-vs-stuffer cosine check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

# Make ``src/`` importable when run as a plain script (no install step).
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from caliber import config  # noqa: E402
from caliber.embeddings import encode_candidates, encode_texts  # noqa: E402
from caliber.index import build_index, save_index  # noqa: E402


def stream_records(path: Path, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Yield candidate records from a JSONL file, one decoded line at a time."""
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def _peak_rss_gb() -> Optional[float]:
    """Peak resident set size in GB (Linux/mac), or None if unavailable."""
    try:
        import resource
    except ImportError:  # non-unix
        return None
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is KB on Linux, bytes on macOS. Heuristic: treat large as bytes.
    kb = ru / 1024 if ru > 1 << 30 else ru
    return kb / (1024 * 1024)


def _shard_manifest(shard_size: int) -> Dict[str, Any]:
    """Parameters a resumed run must match so checkpoints stay consistent."""
    return {
        "shard_size": shard_size,
        "model": config.EMBED_MODEL_NAME,
        "max_seq_length": config.EMBED_MAX_SEQ_LENGTH,
        "dim": config.EMBED_DIM,
    }


def _check_manifest(shard_dir: Path, shard_size: int) -> None:
    """Guard against resuming with mismatched shard params (would corrupt order)."""
    path = shard_dir / "MANIFEST.json"
    want = _shard_manifest(shard_size)
    if path.exists():
        have = json.loads(path.read_text())
        if have != want:
            raise SystemExit(
                f"[precompute] shard manifest mismatch in {shard_dir}:\n"
                f"  existing: {have}\n  requested: {want}\n"
                "Delete the shard dir to re-encode from scratch, or match the params."
            )
    else:
        shard_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(want, indent=2))


def build_embeddings_and_index(
    candidates_path: Path,
    out_dir: Path,
    batch_size: int,
    limit: Optional[int],
    shard_size: int = 5000,
) -> None:
    """Steps 2 + 3: encode all candidates (resumably) and build the FAISS index.

    Candidates are processed in fixed-size shards in strict file order. Each shard
    is checkpointed to ``out_dir/_emb_shards/{emb,ids}_NNNNN.npy`` as it finishes,
    so an interrupted run (this box is slow — a full pass is hours) resumes from
    the first missing shard instead of restarting. Shard boundaries are
    deterministic, so row order — the join key across ``candidate_emb.npy``,
    ``candidate_ids.npy`` and the FAISS index — is identical on every run.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = out_dir / "_emb_shards"
    _check_manifest(shard_dir, shard_size)

    print(f"[precompute] encoding candidates from {candidates_path} "
          f"(shard_size={shard_size}, max_seq_length={config.EMBED_MAX_SEQ_LENGTH}) ...")

    buffer: List[Dict[str, Any]] = []
    shard_idx = 0
    encoded = 0          # candidates actually encoded this run (excludes resumed)
    encode_secs = 0.0    # wall time spent encoding this run

    def flush_shard(records: List[Dict[str, Any]], idx: int) -> None:
        nonlocal encoded, encode_secs
        emb_path = shard_dir / f"emb_{idx:05d}.npy"
        ids_path = shard_dir / f"ids_{idx:05d}.npy"
        if emb_path.exists() and ids_path.exists():
            print(f"[precompute]   shard {idx:05d}: resume (already done, "
                  f"{len(records)} candidates)")
            return
        ids = [r["candidate_id"] for r in records]
        t0 = time.time()
        emb = encode_candidates(records, batch_size=batch_size)
        dt = time.time() - t0
        if emb.shape[0] != len(ids):
            raise RuntimeError(f"shard {idx}: {emb.shape[0]} emb vs {len(ids)} ids")
        # Write ids first, emb last: emb existing is the completion marker, so a
        # crash mid-write never leaves a half-written shard that looks complete.
        np.save(ids_path, np.array(ids))
        np.save(emb_path, emb)
        encoded += emb.shape[0]
        encode_secs += dt
        rate = encoded / encode_secs if encode_secs else 0.0
        print(f"[precompute]   shard {idx:05d}: encoded {emb.shape[0]} in {dt:.1f}s "
              f"| run total {encoded:,} @ {rate:.1f}/s")

    for rec in stream_records(candidates_path, limit):
        buffer.append(rec)
        if len(buffer) >= shard_size:
            flush_shard(buffer, shard_idx)
            buffer = []
            shard_idx += 1
    if buffer:
        flush_shard(buffer, shard_idx)
        shard_idx += 1

    _finalize(out_dir, shard_dir, shard_idx, encoded, encode_secs)


def _finalize(
    out_dir: Path, shard_dir: Path, n_shards: int, encoded: int, encode_secs: float
) -> None:
    """Concatenate shards (in order) into the final artifacts + FAISS index."""
    emb_parts: List[np.ndarray] = []
    id_parts: List[np.ndarray] = []
    for idx in range(n_shards):
        emb_parts.append(np.load(shard_dir / f"emb_{idx:05d}.npy"))
        id_parts.append(np.load(shard_dir / f"ids_{idx:05d}.npy"))

    emb = np.ascontiguousarray(np.vstack(emb_parts), dtype=np.float32)
    ids_arr = np.concatenate(id_parts)
    if emb.shape[0] != ids_arr.shape[0]:
        raise RuntimeError(f"row/id mismatch: {emb.shape[0]} emb vs {ids_arr.shape[0]} ids")

    emb_path = out_dir / config.CANDIDATE_EMB_FILE
    ids_path = out_dir / config.CANDIDATE_IDS_FILE
    idx_path = out_dir / config.FAISS_INDEX_FILE
    np.save(emb_path, emb)
    np.save(ids_path, ids_arr)
    save_index(build_index(emb), idx_path)

    peak = _peak_rss_gb()
    print(f"[precompute] finalized {emb.shape[0]:,} candidates -> {emb.shape} "
          f"{emb.dtype} from {n_shards} shard(s)")
    print(f"[precompute] wrote {emb_path.name}, {ids_path.name}, {idx_path.name} "
          f"to {out_dir}")
    if peak is not None:
        print(f"[precompute] peak RSS: {peak:.2f} GB")
    print(f"[precompute] embed_encode_seconds={encode_secs:.1f} this run "
          f"({encoded:,} newly encoded; record cumulative time in submission_metadata.yaml)")


# --- sanity check ----------------------------------------------------------
# A rough signal that the embeddings are meaningful (NOT a final ranking): a
# genuine ML engineer should sit closer to the JD than a non-tech keyword-stuffer
# who merely lists AI skills. Run against the small sample file so it's fast.

_JD_QUERY = (
    "Senior AI Engineer building and shipping production retrieval, ranking, "
    "recommendation, search, and NLP systems with large language models, "
    "embeddings, and vector databases. Strong applied machine learning "
    "background with real deployed systems at a product company."
)
_ML_TITLE_HINTS = ("machine learning", "ml engineer", "ai engineer", "data scientist",
                   "nlp", "applied scientist", "research engineer", "recommendation",
                   "ranking", "search engineer", "retrieval")
_STUFFER_TITLE_HINTS = ("hr", "human resresource", "human resource", "recruit",
                        "content", "writer", "designer", "graphic", "sales",
                        "accountant", "marketing")
_AI_SKILL_HINTS = ("llm", "rag", "nlp", "machine learning", "deep learning",
                   "transformer", "fine-tuning", "ai", "pytorch", "tensorflow")


def _has_ai_skills(rec: Dict[str, Any]) -> bool:
    names = " ".join((s.get("name") or "").lower() for s in rec.get("skills") or [])
    return any(h in names for h in _AI_SKILL_HINTS)


def _load_records(path: Path) -> List[Dict[str, Any]]:
    """Load candidate records from either a JSON array or a JSONL file."""
    text = path.read_text(encoding="utf-8").lstrip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def run_sanity(sample_path: Path) -> None:
    records = _load_records(sample_path)
    ml = next((r for r in records
               if any(h in (r["profile"].get("current_title") or "").lower()
                      for h in _ML_TITLE_HINTS)), None)
    stuffer = next((r for r in records
                    if any(h in (r["profile"].get("current_title") or "").lower()
                           for h in _STUFFER_TITLE_HINTS) and _has_ai_skills(r)), None)

    if ml is None or stuffer is None:
        print("[sanity] could not find both an ML candidate and a keyword-stuffer "
              f"in {sample_path} (ml={ml is not None}, stuffer={stuffer is not None})")
        return

    jd_vec = encode_texts([_JD_QUERY], is_query=True)[0]
    cand_vecs = encode_candidates([ml, stuffer])
    ml_sim, stuffer_sim = (cand_vecs @ jd_vec).tolist()

    print("[sanity] cosine-to-JD (rough signal, not a ranking):")
    print(f"  ML        {ml['candidate_id']}  "
          f"{ml['profile'].get('current_title')!r:42}  {ml_sim:.4f}")
    print(f"  stuffer   {stuffer['candidate_id']}  "
          f"{stuffer['profile'].get('current_title')!r:42}  {stuffer_sim:.4f}")
    verdict = "PASS" if ml_sim > stuffer_sim else "FAIL"
    print(f"  -> ML closer than stuffer? {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=config.CANDIDATES_PATH)
    parser.add_argument("--out", type=Path, default=config.ARTIFACTS_DIR)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--shard-size", type=int, default=5000,
                        help="candidates per resumable checkpoint shard")
    parser.add_argument("--limit", type=int, default=None,
                        help="encode only the first N records (smoke testing)")
    parser.add_argument("--sanity", action="store_true",
                        help="run the ML-vs-stuffer cosine check and exit")
    parser.add_argument("--sample", type=Path,
                        default=config.DATA_DIR / "challenge" / "sample_candidates.json",
                        help="(sanity only) sample file; JSON array or JSONL")
    args = parser.parse_args()

    if args.sanity:
        run_sanity(args.sample)
        return

    build_embeddings_and_index(args.candidates, args.out, args.batch_size,
                               args.limit, shard_size=args.shard_size)


if __name__ == "__main__":
    main()
