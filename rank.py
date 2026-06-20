#!/usr/bin/env python3
"""Caliber — ONLINE ranking CLI. The documented reproduce entry point.

    python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv

This is the only command Stage-3 runs to reproduce our submission, so it is held
to the hard constraints (CLAUDE.md): ≤ 5 min wall-clock, ≤ 16 GB RAM, CPU-only,
ZERO network, fully deterministic. It does the cheap online half of the two-phase
pipeline and NOTHING expensive:

    1. load the PRECOMPUTED artifacts (jd_profile.json, candidate_ids.npy,
       faiss.index) built offline by scripts/precompute.py — it never re-encodes
       the 100K pool at rank time (that would blow the budget);
    2. stream candidates.jsonl once into an id→Candidate map (memory-safe);
    3. scorer.score_candidates over the pool — default hand-weights (combine);
       LTR stays dormant; the cross-encoder reranks only the shortlist head and
       degrades gracefully if its local model is absent;
    4. ranker.build_submission_rows → the top-100 rows. The honeypot guardrail and
       every DQ-grade invariant (exactly 100, non-increasing score, candidate_id
       tie-break, contiguous ranks, no dupes) are asserted HERE, before any file
       is written;
    5. write the CSV at the exact header + SCORE_DECIMALS precision;
    6. SELF-VALIDATE the written file with tests/validate_submission.py and only
       declare success if it passes;
    7. print load/score/total timing and the honeypot-in-top-100 count, so the big
       run shows the budget headroom and the guardrail margin.

Network is hard-locked offline (HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE) before any
model import, mirroring embeddings.load_model — a stray fetch errors out rather
than silently hitting the hub (a Stage-3 disqualifier).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

# Lock the network OFF before anything below imports torch / sentence-transformers
# (defense in depth; embeddings.load_model does the same). The online path must
# make zero network calls.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Make ``src/`` importable when run as a plain script (no install step needed),
# exactly like scripts/precompute.py.
_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from caliber import config  # noqa: E402
from caliber import ranker  # noqa: E402
from caliber.io_utils import stream_candidates  # noqa: E402
from caliber.scorer import load_jd_profile_artifact, score_candidates  # noqa: E402


# --------------------------------------------------------------------------- #
# Artifact loading (the offline → online handoff). Lazy numpy/faiss imports keep
# the module importable in environments that only need the pure helpers.
# --------------------------------------------------------------------------- #
class ArtifactError(RuntimeError):
    """Raised when a required precompute artifact is missing or inconsistent. The
    message always tells the user to run scripts/precompute.py — we NEVER fall back
    to re-encoding the 100K pool at rank time (that breaks the 5-min budget)."""


def load_artifacts(artifacts_dir: Path) -> tuple[dict, list[str], Any]:
    """Load (jd_profile, candidate_ids, faiss_index) from ``artifacts_dir``.

    candidate_emb.npy is NOT loaded into memory: the FAISS IndexFlatIP already
    holds the candidate vectors, and score_candidates only encodes the JD aspect
    queries at runtime — so the embeddings file is redundant for scoring. When it
    is present we mmap it (no RAM cost) purely to assert row/id/index counts agree,
    catching a stale or half-built artifact set before we score.
    """
    import numpy as np
    from caliber.index import load_index

    artifacts_dir = Path(artifacts_dir)
    jd_path = artifacts_dir / config.JD_PROFILE_FILE
    ids_path = artifacts_dir / config.CANDIDATE_IDS_FILE
    index_path = artifacts_dir / config.FAISS_INDEX_FILE

    missing = [p.name for p in (jd_path, ids_path, index_path) if not p.exists()]
    if missing:
        raise ArtifactError(
            f"Missing required artifact(s) in {artifacts_dir}: {', '.join(missing)}.\n"
            "rank.py loads PRECOMPUTED artifacts and will not re-encode the 100K "
            "pool at rank time (that would break the 5-min budget). Build them once "
            "offline first:\n"
            "    CALIBER_ALLOW_MODEL_DOWNLOAD=1 python scripts/precompute.py"
        )

    jd_profile = load_jd_profile_artifact(jd_path)
    ids_arr = np.load(ids_path, allow_pickle=True)
    candidate_ids = [str(x) for x in ids_arr.tolist()]
    index = load_index(index_path)

    # Consistency guards — fail loudly on a mismatched/stale artifact set.
    ntotal = int(getattr(index, "ntotal", len(candidate_ids)))
    if ntotal != len(candidate_ids):
        raise ArtifactError(
            f"Artifact mismatch: faiss.index has {ntotal} vectors but "
            f"candidate_ids.npy has {len(candidate_ids)} ids. Re-run precompute."
        )
    emb_path = artifacts_dir / config.CANDIDATE_EMB_FILE
    if emb_path.exists():
        emb = np.load(emb_path, mmap_mode="r")  # mmap → shape only, no RAM load
        if emb.shape[0] != len(candidate_ids):
            raise ArtifactError(
                f"Artifact mismatch: candidate_emb.npy has {emb.shape[0]} rows but "
                f"candidate_ids.npy has {len(candidate_ids)} ids. Re-run precompute."
            )

    return jd_profile, candidate_ids, index


# --------------------------------------------------------------------------- #
# CSV writing + self-validation.
# --------------------------------------------------------------------------- #
def write_submission(rows, out_path: Path) -> None:
    """Write the submission CSV: exact header (ranker.SUBMISSION_COLUMNS), score at
    SCORE_DECIMALS precision (the same value the ranker sorted on, so the printed
    bytes keep the non-increasing + tie-break guarantees), reasoning quoted by the
    csv module. UTF-8, '\\n' line terminator."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = f"{{:.{ranker.SCORE_DECIMALS}f}}"
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(ranker.SUBMISSION_COLUMNS)
        for r in rows:
            writer.writerow([r.candidate_id, r.rank, fmt.format(r.score), r.reasoning])


def _load_validator() -> Callable[[Any], list]:
    """Import ``validate_submission`` from tests/validate_submission.py (the exact
    validator the portal uses) without putting tests/ on the path permanently."""
    path = config.ROOT / "tests" / "validate_submission.py"
    spec = importlib.util.spec_from_file_location("_caliber_validate_submission", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ArtifactError(f"could not load the validator at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.validate_submission


def self_validate(out_path: Path) -> list:
    """Run the official format validator on the written CSV; return its error list
    (empty == valid)."""
    return list(_load_validator()(str(out_path)))


# --------------------------------------------------------------------------- #
# The testable core: produce the submission from inputs that may be loaded from
# disk OR injected (tests pass a tiny on-the-fly pool + fake retrieval seams, like
# eval/evaluate.py, so no model / no 100K / no faiss is needed).
# --------------------------------------------------------------------------- #
def produce_submission(
    *,
    candidates_path: Any = None,
    artifacts_dir: Any = None,
    out_path: Any = "submission.csv",
    top_n: int = ranker.TOP_N,
    ce_enabled: bool = True,
    do_self_validate: bool = True,
    # injection seams (default = load from disk / real components):
    jd_profile: Optional[Mapping[str, Any]] = None,
    candidate_ids: Optional[list] = None,
    faiss_index: Any = None,
    candidates_by_id: Optional[Mapping[str, Any]] = None,
    score_fn: Optional[Callable] = None,
    encode_query_fn: Optional[Callable] = None,
    search_fn: Optional[Callable] = None,
    rerank_fn: Optional[Callable] = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Build the top-``top_n`` submission CSV and self-validate it.

    Returns a dict of timing + counts (``load_seconds``, ``score_seconds``,
    ``total_seconds``, ``n_pool``, ``n_scored``, ``honeypots_in_top``,
    ``out_path``, ``validation_errors``). Raises ``ranker.HoneypotGuardrailError``
    (no CSV written) if the top-``top_n`` would breach the honeypot guardrail, and
    ``ArtifactError`` if required artifacts are missing.

    Artifacts (jd_profile / candidate_ids / faiss_index) are loaded from
    ``artifacts_dir`` unless injected. The candidate pool is streamed from
    ``candidates_path`` unless ``candidates_by_id`` is injected. All ``*_fn`` seams
    default to the scorer's real components.
    """
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else config.ARTIFACTS_DIR
    candidates_path = candidates_path if candidates_path is not None else config.CANDIDATES_PATH
    out_path = Path(out_path)

    t0 = time.perf_counter()

    # 1 — artifacts (unless fully injected).
    if jd_profile is None or candidate_ids is None or faiss_index is None:
        log(f"[rank] loading artifacts from {artifacts_dir} ...")
        loaded_jd, loaded_ids, loaded_index = load_artifacts(artifacts_dir)
        jd_profile = jd_profile if jd_profile is not None else loaded_jd
        candidate_ids = candidate_ids if candidate_ids is not None else loaded_ids
        faiss_index = faiss_index if faiss_index is not None else loaded_index
    candidate_ids = [str(x) for x in candidate_ids]

    # 2 — stream the pool once into an id→Candidate map (unless injected). We build
    # it here so the SAME typed Candidates feed both the scorer and reasoning (the
    # title+years lead) — one pass, no double read of the 465 MB file.
    if candidates_by_id is None:
        log(f"[rank] streaming candidates from {candidates_path} ...")
        candidates_by_id = {str(c.candidate_id): c for c in stream_candidates(candidates_path)}
    else:
        candidates_by_id = {str(k): v for k, v in candidates_by_id.items()}
    n_pool = len(candidates_by_id)
    load_seconds = time.perf_counter() - t0
    log(f"[rank] loaded {n_pool:,} candidates + artifacts in {load_seconds:.1f}s")

    # 3 — score the pool (default hand-weights / combine; CE on the shortlist head).
    log(f"[rank] scoring (ce_enabled={ce_enabled}) ...")
    t_score = time.perf_counter()
    scorer_kwargs: dict[str, Any] = dict(
        jd_profile=jd_profile,
        candidate_ids=candidate_ids,
        faiss_index=faiss_index,
        candidates_by_id=candidates_by_id,
        ce_enabled=ce_enabled,
    )
    for name, fn in (("encode_query_fn", encode_query_fn), ("search_fn", search_fn),
                     ("rerank_fn", rerank_fn)):
        if fn is not None:
            scorer_kwargs[name] = fn
    results = (score_fn or score_candidates)(**scorer_kwargs)
    score_seconds = time.perf_counter() - t_score
    log(f"[rank] scored {len(results):,} shortlisted candidates in {score_seconds:.1f}s")

    # 4 — select top-N + invariants + guardrail + grounded reasoning (raises here on
    # any violation, BEFORE we touch the filesystem).
    rows = ranker.build_submission_rows(results, top_n=top_n, candidates=candidates_by_id)

    # Honeypot-in-top-N count (must be 0; guardrail already enforces < 10%).
    top_ids = {r.candidate_id for r in rows}
    honeypots_in_top = sum(
        1 for cid in top_ids if getattr(results.get(cid), "is_honeypot", False)
    )

    # 5 — write the CSV.
    write_submission(rows, out_path)
    log(f"[rank] wrote {len(rows)} rows → {out_path}")

    # 6 — self-validate the written file.
    validation_errors: list = []
    if do_self_validate:
        validation_errors = self_validate(out_path)

    total_seconds = time.perf_counter() - t0
    return {
        "out_path": str(out_path),
        "n_pool": n_pool,
        "n_scored": len(results),
        "n_rows": len(rows),
        "honeypots_in_top": honeypots_in_top,
        "load_seconds": load_seconds,
        "score_seconds": score_seconds,
        "total_seconds": total_seconds,
        "validation_errors": validation_errors,
    }


def _print_summary(out: dict, top_n: int) -> None:
    print("")
    print("=" * 64)
    print("CALIBER — submission summary")
    print("=" * 64)
    print(f"  output file              : {out['out_path']}")
    print(f"  pool size                : {out['n_pool']:,}")
    print(f"  shortlisted / scored     : {out['n_scored']:,}")
    print(f"  rows written             : {out['n_rows']}")
    print(f"  honeypots in top-{top_n:<3}     : {out['honeypots_in_top']}  "
          f"(guardrail: < {int(ranker.HONEYPOT_MAX_FRACTION * 100)}%)")
    print(f"  load time                : {out['load_seconds']:.1f}s")
    print(f"  score time               : {out['score_seconds']:.1f}s")
    print(f"  total time               : {out['total_seconds']:.1f}s  "
          f"(budget: 300s)")
    errs = out["validation_errors"]
    print("-" * 64)
    if errs:
        print(f"  SELF-VALIDATION: FAIL ({len(errs)} issue(s)):")
        for e in errs:
            print(f"    - {e}")
    else:
        print("  SELF-VALIDATION: PASS ✓  (matches validate_submission.py)")
    print("=" * 64)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", type=Path, default=config.CANDIDATES_PATH,
                        help="path to candidates.jsonl(.gz) (default: data/candidates.jsonl)")
    parser.add_argument("--artifacts", type=Path, default=config.ARTIFACTS_DIR,
                        help="precompute artifacts dir (jd_profile.json, candidate_ids.npy, "
                             "faiss.index)")
    parser.add_argument("--out", type=Path, default=Path("submission.csv"),
                        help="output CSV path (rename to <participant_id>.csv before upload)")
    parser.add_argument("--no-ce", action="store_true",
                        help="disable the cross-encoder rerank (it degrades gracefully "
                             "if the local model is absent anyway)")
    parser.add_argument("--no-validate", action="store_true",
                        help="skip the self-validation step (not recommended)")
    args = parser.parse_args(argv)

    try:
        out = produce_submission(
            candidates_path=args.candidates,
            artifacts_dir=args.artifacts,
            out_path=args.out,
            ce_enabled=not args.no_ce,
            do_self_validate=not args.no_validate,
        )
    except ranker.HoneypotGuardrailError as e:
        print(f"\n[rank] ABORTED — honeypot guardrail tripped, no CSV written:\n{e}",
              file=sys.stderr)
        return 2
    except ArtifactError as e:
        print(f"\n[rank] ERROR — {e}", file=sys.stderr)
        return 2

    _print_summary(out, ranker.TOP_N)
    # Non-zero exit if the self-validation found problems, so CI / the reproduce
    # step fails loudly rather than shipping a malformed CSV.
    return 1 if out["validation_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
