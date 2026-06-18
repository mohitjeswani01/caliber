"""OFFLINE harness — score the silver set with the REAL scorer, report metrics.

Glue between ``src/caliber/scorer.py`` (the online composite scorer) and
``eval/metrics.py`` (the official composite). It answers the only question that
matters before we spend one of our 3 submissions: *how well does our scorer rank
the candidates whose true relevance we know?*

The measurement (STRATEGY.md §7), kept deliberately clean:

  1. Load the silver answer key -> ``{candidate_id: grade_final}`` (0-4), skipping
     ungraded / needs-review records.
  2. Stream ``candidates.jsonl`` and keep ONLY the silver candidates' full records
     (typed ``Candidate`` objects). These carry NO grade field — the scorer never
     sees a label.
  3. Build a **silver-only** bge embedding matrix + FAISS ``IndexFlatIP`` over just
     those ~400 candidates (no 100K precompute needed). Cosine is independent of
     what else is in the index, and the per-aspect retrieval depth covers the whole
     set, so each candidate gets exactly the semantic_sim the full pipeline would
     assign it.
  4. Run ``scorer.score_candidates`` over them -> a ``final_score`` each.
  5. Rank by ``final_score`` desc, ``candidate_id`` asc (the scorer's own
     deterministic contract), and hand THAT ranking to
     ``metrics.evaluate_ranking`` together with the grades.

No labels ever enter the scoring path: the scorer is given candidate data and a
JD profile; the eval compares its produced ranking to the grades *afterward*.

This module is **eval-only**. It must NEVER be imported by ``rank.py`` or any
``src/caliber`` online module (that would be training the ranker on its own test
set). It imports the scorer read-only, the same way the real online path will.

Run:
    python eval/evaluate.py                       # silver-only index, CE off
    python eval/evaluate.py --ce                  # turn the cross-encoder on
    python eval/evaluate.py --threshold 1.0       # looser binary diagnostic

Ablation hook: ``evaluate_silver`` takes ``ce_enabled``, ``weights`` and
``combine_fn`` so a later sweep can call it repeatedly (CE on/off, weight sets,
LTR ``predict`` as ``combine_fn``) and tabulate the composite for the deck. Each
of those is a pure pass-through to ``score_candidates`` — nothing here hardcodes
a single configuration.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

# Repo root on sys.path so ``eval`` and ``caliber`` resolve regardless of CWD
# (mirrors scripts/make_silver_labels.py; conftest.py does the same for tests).
ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval.metrics import evaluate_ranking  # noqa: E402
from caliber import config  # noqa: E402  (SEED / paths; import-safe, no torch/faiss)

# Default locations (all overridable on the CLI).
DEFAULT_SILVER_LABELS = ROOT / "eval" / "silver_labels.json"
DEFAULT_CANDIDATES = ROOT / "data" / "candidates.jsonl"
DEFAULT_JD_PROFILE = ROOT / "artifacts" / "jd_profile.json"
# Hand-judged ANCHOR grades, merged ON TOP of the silver labels. These are
# correct labels assigned by reading the profile against the STRATEGY §4 rubric
# (never by a scorer's rank), used to grade genuine fits that the random pool
# surfaced but the stratified silver sample never covered. Treated as truth — an
# anchor, never tuned against. Optional: missing file => no overlay.
DEFAULT_MANUAL_GRADES = ROOT / "eval" / "manual_grades.json"

# Default size of the random UNLABELED draw for the realistic-pool mode. ~8000
# distractors against the 400 silver candidates approximates the real task —
# surfacing the few genuine fits out of a ~99%-irrelevant sea. Configurable on
# the CLI so a fast first read can start smaller (e.g. --pool-size 3000).
DEFAULT_POOL_SIZE = 8000


# --------------------------------------------------------------------------- #
# Step 1 — load the silver answer key.
# --------------------------------------------------------------------------- #
def load_grades(path: Any = DEFAULT_SILVER_LABELS) -> dict[str, float]:
    """Read ``silver_labels.json`` -> ``{candidate_id: grade_final}`` as floats.

    Skips any record whose ``grade_final`` is ``None`` or that is flagged
    ``needs_review`` — those are exactly the entries the silver pipeline could not
    confidently grade, so including them as ground truth would be noise. Grades
    are the ONLY thing read here; nothing else about the record is kept.
    """
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    grades: dict[str, float] = {}
    for rec in records:
        if rec.get("needs_review"):
            continue
        g = rec.get("grade_final")
        if g is None:
            continue
        grades[str(rec["candidate_id"])] = float(g)
    return grades


def load_manual_grades(path: Any = DEFAULT_MANUAL_GRADES) -> dict[str, float]:
    """Read the hand-judged ANCHOR overlay -> ``{candidate_id: grade}`` as floats.

    The overlay is a JSON list of records, each ``{"candidate_id": ..., "grade":
    ...}`` (extra fields like ``rationale`` / ``source`` are documentation and are
    ignored here). A record missing either key is skipped. Returns ``{}`` if the
    file does not exist — the overlay is strictly optional, so the eval runs
    identically on a checkout that has none.

    These are correct labels (graded by profile against the STRATEGY §4 rubric),
    NOT predictions — they are merged on top of the silver labels and treated as
    ground truth, never tuned against.
    """
    p = Path(path)
    if not p.exists():
        return {}
    records = json.loads(p.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for rec in records:
        cid = rec.get("candidate_id")
        g = rec.get("grade")
        if cid is None or g is None:
            continue
        out[str(cid)] = float(g)
    return out


def resolve_grades(
    silver_labels_path: Any = DEFAULT_SILVER_LABELS,
    manual_grades_path: Any = DEFAULT_MANUAL_GRADES,
) -> tuple[dict[str, float], int]:
    """Load silver grades and merge the manual anchor overlay ON TOP, returning
    ``(merged_grades, n_manual_anchors)``.

    Manual anchors OVERRIDE the silver grade for the same id (so an anchor can
    correct a label), and add ids the silver set never sampled. The silver set is
    NOT regenerated or re-sampled — the overlay is layered at grade-lookup time, so
    anchor ids that were never in the stratified silver draw are still honored.
    """
    silver = load_grades(silver_labels_path)
    manual = load_manual_grades(manual_grades_path)
    merged = {**silver, **manual}  # manual wins on key collision
    return merged, len(manual)


# --------------------------------------------------------------------------- #
# Step 2 — pull the silver candidates' full records (no grades attached).
# --------------------------------------------------------------------------- #
def load_silver_candidates(candidates_path: Any, ids: set[str]) -> dict[str, Any]:
    """Stream ``candidates.jsonl`` once and return ``{id: Candidate}`` for ``ids``.

    Uses ``io_utils.stream_candidates`` (memory-safe, one record at a time) and
    stops as soon as every requested id is found, so it never scans the whole
    465 MB file when the silver set is small. The returned objects are typed
    ``Candidate``s straight from ``parse_candidate`` — they contain candidate
    facts only, never a relevance grade.
    """
    from caliber.io_utils import stream_candidates

    want = set(ids)
    found: dict[str, Any] = {}
    for cand in stream_candidates(candidates_path):
        cid = str(cand.candidate_id)
        if cid in want:
            found[cid] = cand
            if len(found) == len(want):
                break
    return found


# --------------------------------------------------------------------------- #
# Step 2b (REALISTIC MODE) — build the harder pool: silver + a deterministic
# random draw of UNLABELED candidates.
# --------------------------------------------------------------------------- #
def sample_realistic_pool(
    candidates_path: Any,
    silver_ids: set[str],
    *,
    pool_size: int,
    seed: int = config.SEED,
) -> tuple[dict[str, Any], int, int]:
    """Stream ``candidates.jsonl`` ONCE and return a combined ``{id: Candidate}``
    map plus ``(n_silver_found, n_unlabeled_sampled)``.

    The combined pool is the silver candidates (those in ``silver_ids``, our only
    ground truth) PLUS a uniform random sample of ``pool_size`` candidates that are
    NOT in the silver set. The added candidates are intentionally UNLABELED — the
    metrics treat any id absent from the grade map as relevance 0 (a random draw
    from a ~99%-irrelevant pool is almost entirely true-negatives, so grade 0 is
    the honest default). There is a negligible risk the draw contains an unlabeled
    genuine fit; at ``pool_size``≈8000 against rare real fits we accept it — it can
    only understate the score, never inflate it.

    Sampling is **reservoir sampling (Algorithm R)** driven by a fresh
    ``random.Random(seed)`` over the fixed non-silver stream order, so it is fully
    deterministic (same seed + same file ⇒ identical sample) AND memory-safe: it
    streams the 465 MB file one record at a time, holding at most
    ``pool_size`` + |silver| typed records, never the whole pool.
    """
    from caliber.io_utils import stream_candidates

    rng = random.Random(seed)
    want = set(silver_ids)
    silver_found: dict[str, Any] = {}
    reservoir: list[Any] = []
    seen_unlabeled = 0  # count of non-silver records seen so far == current item's 0-based index

    for cand in stream_candidates(candidates_path):
        cid = str(cand.candidate_id)
        if cid in want:
            silver_found[cid] = cand
            continue
        if pool_size <= 0:
            continue
        # Reservoir sampling: fill, then replace with probability pool_size/i.
        if len(reservoir) < pool_size:
            reservoir.append(cand)
        else:
            j = rng.randint(0, seen_unlabeled)  # inclusive 0..i
            if j < pool_size:
                reservoir[j] = cand
        seen_unlabeled += 1

    combined = dict(silver_found)
    for cand in reservoir:
        combined[str(cand.candidate_id)] = cand
    return combined, len(silver_found), len(reservoir)


# --------------------------------------------------------------------------- #
# Step 3 — build a silver-only embedding index on the fly.
# --------------------------------------------------------------------------- #
def build_silver_index(
    ordered_ids: Sequence[str],
    candidates_by_id: Mapping[str, Any],
    encode_candidates_fn: Optional[Callable] = None,
):
    """Encode the silver candidates (in ``ordered_ids`` order) and build a FAISS
    ``IndexFlatIP`` over them. Returns ``(index, ordered_ids)`` where row ``i`` of
    the index is ``ordered_ids[i]`` — the join key ``score_candidates`` relies on.

    ``encode_candidates_fn`` defaults to ``embeddings.encode_candidates`` (the same
    bge model the online path uses); tests inject a light fake so no model/torch is
    needed. Imports are lazy so importing this module pulls in no faiss/torch.
    """
    if encode_candidates_fn is None:
        from caliber.embeddings import encode_candidates as encode_candidates_fn  # type: ignore
    from caliber.index import build_index

    cands = [candidates_by_id[cid] for cid in ordered_ids]
    emb = encode_candidates_fn(cands)
    index = build_index(emb)
    return index, list(ordered_ids)


# --------------------------------------------------------------------------- #
# Steps 4-5 — score the silver set and turn it into a ranking.
# --------------------------------------------------------------------------- #
def rank_silver(
    candidates_by_id: Mapping[str, Any],
    jd_profile: Mapping[str, Any],
    *,
    ce_enabled: bool = False,
    weights: Optional[Mapping[str, float]] = None,
    combine_fn: Optional[Callable] = None,
    score_fn: Optional[Callable] = None,
    encode_candidates_fn: Optional[Callable] = None,
    encode_query_fn: Optional[Callable] = None,
    search_fn: Optional[Callable] = None,
    rerank_fn: Optional[Callable] = None,
) -> tuple[list[str], dict[str, Any]]:
    """Score the silver candidates and return ``(ranked_ids, results)``.

    ``ranked_ids`` is sorted by ``final_score`` descending, ``candidate_id``
    ascending — the exact deterministic contract ``ranker.py`` enforces, so this
    measures the ordering the real submission would produce. Any silver candidate
    the scorer did not return (it shouldn't happen — retrieval depth covers the
    whole set) is appended deterministically at the end so the ranking still
    covers every candidate we have a record for.

    The ``*_fn`` / ``weights`` / ``combine_fn`` / ``ce_enabled`` params are the
    ablation + test seams: pass-throughs to ``score_candidates`` (or a fake
    ``score_fn``). ``None`` lets ``score_candidates`` resolve its own defaults.
    """
    if score_fn is None:
        from caliber.scorer import score_candidates as score_fn  # type: ignore

    ordered_ids = sorted(candidates_by_id)  # deterministic row order
    index, candidate_ids = build_silver_index(
        ordered_ids, candidates_by_id, encode_candidates_fn
    )

    # Only forward optional seams that were actually supplied, so score_candidates
    # falls back to its own documented defaults (embeddings/index/cross_encoder)
    # for anything we don't override.
    kwargs: dict[str, Any] = dict(
        jd_profile=jd_profile,
        candidate_ids=candidate_ids,
        faiss_index=index,
        candidates_by_id=candidates_by_id,
        ce_enabled=ce_enabled,
    )
    if weights is not None:
        kwargs["weights"] = weights
    if combine_fn is not None:
        kwargs["combine_fn"] = combine_fn
    if encode_query_fn is not None:
        kwargs["encode_query_fn"] = encode_query_fn
    if search_fn is not None:
        kwargs["search_fn"] = search_fn
    if rerank_fn is not None:
        kwargs["rerank_fn"] = rerank_fn

    results = score_fn(**kwargs)

    ranked = sorted(
        results.values(), key=lambda cs: (-cs.final_score, cs.candidate_id)
    )
    ranked_ids = [cs.candidate_id for cs in ranked]
    # Defensive: include any record-bearing silver id the scorer didn't return, so
    # the ranking still spans every candidate we could have ranked. Deterministic.
    missing = sorted(set(map(str, candidates_by_id)) - set(ranked_ids))
    ranked_ids.extend(missing)
    return ranked_ids, results


# --------------------------------------------------------------------------- #
# Reporting + sanity checks (STRATEGY.md §7).
# --------------------------------------------------------------------------- #
def _grade_hist(ids: Sequence[str], grades: Mapping[str, float]) -> dict[int, int]:
    h: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for cid in ids:
        g = int(grades.get(cid, 0))
        h[g] = h.get(g, 0) + 1
    return h


def build_report(
    metrics: Mapping[str, float],
    ranked_ids: Sequence[str],
    results: Mapping[str, Any],
    grades: Mapping[str, float],
    *,
    n_graded: int,
    n_found: int,
    ce_enabled: bool,
    threshold: float,
    mode: str = "silver",
    n_unlabeled: Optional[int] = None,
    seed: Optional[int] = None,
    encode_seconds: Optional[float] = None,
    total_seconds: Optional[float] = None,
    n_manual: int = 0,
) -> str:
    """Format the human-readable eval report (returned as a string; printed by
    the CLI). Pure — no IO — so it is unit-testable.

    ``mode`` selects the header + setup block: ``"silver"`` (the within-set
    diagnostic, default) or ``"realistic"`` (silver + ``n_unlabeled`` random
    unlabeled distractors). The realistic block additionally reports the pool
    composition, the sampling seed and encode/total timing.
    """
    n_scored = len(results)
    ce_used = sum(1 for cs in results.values() if getattr(cs, "ce_used", False))
    honeypots = [cid for cid in ranked_ids if getattr(results.get(cid), "is_honeypot", False)]

    def hp_in(top: int) -> int:
        return sum(1 for cid in ranked_ids[:top] if cid in set(honeypots))

    top10_hist = _grade_hist(ranked_ids[:10], grades)

    lines: list[str] = []
    lines.append("=" * 70)
    if mode == "realistic":
        nun = n_unlabeled or 0
        lines.append(f"CALIBER — REALISTIC POOL (silver + {nun} random) — real scorer vs known grades")
        lines.append("=" * 70)
        lines.append("")
        lines.append("Setup:")
        lines.append(f"  graded silver candidates (ground truth):         {n_graded}")
        if n_manual:
            lines.append(f"  ...of which hand-judged manual anchors:          {n_manual}")
        lines.append(f"  silver candidates found in pool:                 {n_found}")
        lines.append(f"  random UNLABELED candidates (treated grade 0):   {nun}")
        lines.append(f"  combined pool size / scored (shortlist):         {n_found + nun} / {n_scored}")
        lines.append(f"  random sample seed (deterministic):              {seed}")
        lines.append(f"  artifacts:                                       combined bge index (built on the fly)")
        lines.append(f"  cross-encoder:                                   {'ENABLED' if ce_enabled else 'disabled'}"
                     f" (scored {ce_used} candidates)")
        lines.append(f"  binary relevance threshold (MAP / P@10):         tier >= {threshold:g}")
        if encode_seconds is not None or total_seconds is not None:
            et = "n/a" if encode_seconds is None else f"{encode_seconds:.1f}s"
            tt = "n/a" if total_seconds is None else f"{total_seconds:.1f}s"
            lines.append(f"  encode time / total time:                        {et} / {tt}")
        lines.append("")
    else:
        lines.append("CALIBER — SILVER-SET EVALUATION (real scorer vs known grades)")
        lines.append("=" * 70)
        lines.append("")
        lines.append("Setup:")
        lines.append(f"  silver grades loaded (graded, not needs-review): {n_graded}")
        if n_manual:
            lines.append(f"  ...of which hand-judged manual anchors:          {n_manual}")
        lines.append(f"  silver candidates found in pool / scored:        {n_found} / {n_scored}")
        lines.append(f"  artifacts:                                       silver-only bge index (built on the fly)")
        lines.append(f"  cross-encoder:                                   {'ENABLED' if ce_enabled else 'disabled'}"
                     f" (scored {ce_used} candidates)")
        lines.append(f"  binary relevance threshold (MAP / P@10):         tier >= {threshold:g}")
        lines.append("")
    lines.append("Official composite (0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10):")
    lines.append(f"  NDCG@10 : {metrics['ndcg@10']:.4f}   (weight 0.50)")
    lines.append(f"  NDCG@50 : {metrics['ndcg@50']:.4f}   (weight 0.30)")
    lines.append(f"  MAP     : {metrics['map']:.4f}   (weight 0.15)")
    lines.append(f"  P@10    : {metrics['p@10']:.4f}   (weight 0.05)")
    lines.append(f"  -------")
    lines.append(f"  COMPOSITE : {metrics['composite']:.4f}")
    lines.append("")
    lines.append("Sanity checks (STRATEGY.md §7):")
    lines.append(f"  honeypots flagged by scorer (total):  {len(honeypots)}")
    lines.append(f"  honeypots in top 10 / 50 / 100:       {hp_in(10)} / {hp_in(50)} / {hp_in(100)}")
    lines.append(f"  top-10 grade histogram (4..0):        "
                 f"{top10_hist[4]}/{top10_hist[3]}/{top10_hist[2]}/{top10_hist[1]}/{top10_hist[0]}")
    lines.append("")
    lines.append("Top 10 by scorer:")
    for rank, cid in enumerate(ranked_ids[:10], 1):
        cs = results.get(cid)
        g = grades.get(cid)
        gtxt = f"grade={int(g)}" if g is not None else "grade=?"
        score = getattr(cs, "final_score", float("nan"))
        hp = " [HONEYPOT-FLOORED]" if getattr(cs, "is_honeypot", False) else ""
        lines.append(f"  {rank:2d}. {cid}  score={score:+.4f}  {gtxt}{hp}")
    lines.append("=" * 70)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Top-level orchestration.
# --------------------------------------------------------------------------- #
def _metrics_and_threshold(
    ranked_ids: Sequence[str],
    grades: Mapping[str, float],
    threshold: Optional[float],
) -> tuple[dict[str, float], float]:
    """Compute the official composite over ``ranked_ids`` vs ``grades`` and return
    ``(metrics, used_threshold)``. ``threshold=None`` uses the official tier>=3.0
    baked into ``metrics.evaluate_ranking``. Shared by both eval modes so the
    metric path is identical regardless of which pool produced the ranking."""
    if threshold is None:
        from eval.metrics import OFFICIAL_RELEVANCE_THRESHOLD
        return dict(evaluate_ranking(ranked_ids, dict(grades))), OFFICIAL_RELEVANCE_THRESHOLD
    return dict(evaluate_ranking(ranked_ids, dict(grades), threshold=threshold)), threshold


def evaluate_silver(
    *,
    silver_labels_path: Any = DEFAULT_SILVER_LABELS,
    manual_grades_path: Any = DEFAULT_MANUAL_GRADES,
    candidates_path: Any = DEFAULT_CANDIDATES,
    jd_profile: Optional[Mapping[str, Any]] = None,
    jd_profile_path: Any = DEFAULT_JD_PROFILE,
    ce_enabled: bool = False,
    weights: Optional[Mapping[str, float]] = None,
    combine_fn: Optional[Callable] = None,
    threshold: Optional[float] = None,
    # test/ablation seams (all default to the real components):
    score_fn: Optional[Callable] = None,
    encode_candidates_fn: Optional[Callable] = None,
    encode_query_fn: Optional[Callable] = None,
    search_fn: Optional[Callable] = None,
    rerank_fn: Optional[Callable] = None,
    grades: Optional[Mapping[str, float]] = None,
    candidates_by_id: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Run the whole silver-set evaluation and return a result dict::

        {"metrics": {...}, "ranked_ids": [...], "results": {...},
         "report": "<text>", "n_graded": int, "n_found": int}

    ``grades`` / ``candidates_by_id`` may be injected (tests / a pre-built pool);
    otherwise they are loaded from disk — silver labels with the manual anchor
    overlay merged on top (``resolve_grades``). ``jd_profile`` may be injected;
    otherwise it is read from ``jd_profile_path`` via the scorer's own loader (we
    reuse ``scorer.load_jd_profile_artifact`` rather than duplicate a JD loader).
    """
    n_manual = 0
    if grades is None:
        grades, n_manual = resolve_grades(silver_labels_path, manual_grades_path)
    if jd_profile is None:
        from caliber.scorer import load_jd_profile_artifact
        jd_profile = load_jd_profile_artifact(jd_profile_path)
    if candidates_by_id is None:
        candidates_by_id = load_silver_candidates(candidates_path, set(grades))

    n_graded = len(grades)
    n_found = len(candidates_by_id)

    ranked_ids, results = rank_silver(
        candidates_by_id,
        jd_profile,
        ce_enabled=ce_enabled,
        weights=weights,
        combine_fn=combine_fn,
        score_fn=score_fn,
        encode_candidates_fn=encode_candidates_fn,
        encode_query_fn=encode_query_fn,
        search_fn=search_fn,
        rerank_fn=rerank_fn,
    )

    # Default to the OFFICIAL threshold baked into metrics.evaluate_ranking.
    metrics, used_threshold = _metrics_and_threshold(ranked_ids, grades, threshold)

    report = build_report(
        metrics, ranked_ids, results, grades,
        n_graded=n_graded, n_found=n_found,
        ce_enabled=ce_enabled, threshold=used_threshold,
        n_manual=n_manual,
    )
    return {
        "metrics": dict(metrics),
        "ranked_ids": ranked_ids,
        "results": results,
        "report": report,
        "n_graded": n_graded,
        "n_found": n_found,
        "n_manual": n_manual,
    }


def evaluate_realistic(
    *,
    pool_size: int = DEFAULT_POOL_SIZE,
    seed: int = config.SEED,
    silver_labels_path: Any = DEFAULT_SILVER_LABELS,
    manual_grades_path: Any = DEFAULT_MANUAL_GRADES,
    candidates_path: Any = DEFAULT_CANDIDATES,
    jd_profile: Optional[Mapping[str, Any]] = None,
    jd_profile_path: Any = DEFAULT_JD_PROFILE,
    ce_enabled: bool = False,
    weights: Optional[Mapping[str, float]] = None,
    combine_fn: Optional[Callable] = None,
    threshold: Optional[float] = None,
    # test/ablation seams (all default to the real components):
    score_fn: Optional[Callable] = None,
    encode_candidates_fn: Optional[Callable] = None,
    encode_query_fn: Optional[Callable] = None,
    search_fn: Optional[Callable] = None,
    rerank_fn: Optional[Callable] = None,
    grades: Optional[Mapping[str, float]] = None,
    candidates_by_id: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """REALISTIC-POOL evaluation: score the silver candidates AMONG a large random
    draw of unlabeled distractors, and report the official composite using ONLY the
    silver grades (unlabeled = relevance 0).

    This mirrors the real task — find the few genuine fits in a ~99%-irrelevant
    sea — so its composite is the honest baseline to tune against. It is expected
    to be LOWER than the within-set silver number, which over-estimates because the
    400 hand-picked candidates are an easy, relevant-rich separation.

    Returns the same dict shape as ``evaluate_silver`` plus ``n_unlabeled``,
    ``pool_size``, ``encode_seconds`` and ``total_seconds``. Reuses the exact
    ``rank_silver`` → ``build_silver_index`` → ``score_candidates`` path; the only
    new step is building the bigger combined pool (``sample_realistic_pool``).

    ``grades`` / ``candidates_by_id`` may be injected (tests); otherwise the pool
    is sampled from ``candidates_path`` and grades loaded from disk. ``weights`` /
    ``combine_fn`` / ``ce_enabled`` are pure pass-throughs so prompt 2's sweep can
    drive this mode.
    """
    t_start = time.perf_counter()

    n_manual = 0
    if grades is None:
        grades, n_manual = resolve_grades(silver_labels_path, manual_grades_path)
    if jd_profile is None:
        from caliber.scorer import load_jd_profile_artifact
        jd_profile = load_jd_profile_artifact(jd_profile_path)

    if candidates_by_id is None:
        # set(grades) includes the manual anchor ids, so they are force-INCLUDED
        # in the pool (kept, never left to the random draw) and the N unlabeled are
        # sampled from everything else.
        candidates_by_id, n_found, n_unlabeled = sample_realistic_pool(
            candidates_path, set(grades), pool_size=pool_size, seed=seed
        )
    else:
        # Injected pool (tests / a pre-built combination): anything not graded is
        # the unlabeled portion. Found = silver candidates actually present.
        n_found = sum(1 for cid in candidates_by_id if str(cid) in grades)
        n_unlabeled = len(candidates_by_id) - n_found

    # Time the (expensive) candidate encode by wrapping whichever encoder is in
    # play — the injected fake in tests, or the real bge encoder resolved lazily.
    enc_accum = {"t": 0.0}
    base_encode = encode_candidates_fn

    def timed_encode(cands):
        fn = base_encode
        if fn is None:
            from caliber.embeddings import encode_candidates as fn  # type: ignore
        t0 = time.perf_counter()
        emb = fn(cands)
        enc_accum["t"] += time.perf_counter() - t0
        return emb

    ranked_ids, results = rank_silver(
        candidates_by_id,
        jd_profile,
        ce_enabled=ce_enabled,
        weights=weights,
        combine_fn=combine_fn,
        score_fn=score_fn,
        encode_candidates_fn=timed_encode,
        encode_query_fn=encode_query_fn,
        search_fn=search_fn,
        rerank_fn=rerank_fn,
    )

    metrics, used_threshold = _metrics_and_threshold(ranked_ids, grades, threshold)
    total_seconds = time.perf_counter() - t_start

    report = build_report(
        metrics, ranked_ids, results, grades,
        n_graded=len(grades), n_found=n_found,
        ce_enabled=ce_enabled, threshold=used_threshold,
        mode="realistic", n_unlabeled=n_unlabeled, seed=seed,
        encode_seconds=enc_accum["t"], total_seconds=total_seconds,
        n_manual=n_manual,
    )
    return {
        "metrics": dict(metrics),
        "ranked_ids": ranked_ids,
        "results": results,
        "report": report,
        "n_graded": len(grades),
        "n_found": n_found,
        "n_unlabeled": n_unlabeled,
        "n_manual": n_manual,
        "pool_size": pool_size,
        "encode_seconds": enc_accum["t"],
        "total_seconds": total_seconds,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the real scorer against the silver labels (offline)."
    )
    parser.add_argument("--silver-labels", default=str(DEFAULT_SILVER_LABELS))
    parser.add_argument("--manual-grades", default=str(DEFAULT_MANUAL_GRADES),
                        help="hand-judged anchor grades merged on top of the silver "
                             "labels (optional; ignored if the file is absent).")
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--jd-profile", default=str(DEFAULT_JD_PROFILE))
    parser.add_argument("--ce", action="store_true",
                        help="enable the cross-encoder rerank (needs the local CE model)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="binary relevance cutoff for MAP/P@10 "
                             "(default: official tier>=3.0)")
    parser.add_argument("--pool-size", type=int, default=None,
                        help="REALISTIC mode: add this many random UNLABELED "
                             "candidates (grade 0) to the silver set before scoring. "
                             f"Omit for the within-set silver mode; ~{DEFAULT_POOL_SIZE} "
                             "mirrors the real task (start smaller, e.g. 3000, for a "
                             "fast first read).")
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help="random seed for the realistic-mode sample (deterministic).")
    args = parser.parse_args(argv)

    if args.pool_size is not None:
        out = evaluate_realistic(
            pool_size=args.pool_size,
            seed=args.seed,
            silver_labels_path=args.silver_labels,
            manual_grades_path=args.manual_grades,
            candidates_path=args.candidates,
            jd_profile_path=args.jd_profile,
            ce_enabled=args.ce,
            threshold=args.threshold,
        )
    else:
        out = evaluate_silver(
            silver_labels_path=args.silver_labels,
            manual_grades_path=args.manual_grades,
            candidates_path=args.candidates,
            jd_profile_path=args.jd_profile,
            ce_enabled=args.ce,
            threshold=args.threshold,
        )
    print(out["report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
