"""eval/sweep.py — OFFLINE weight sweep over the realistic pool (PROMPT 2).

The methodology is the point; the code just enforces it.

  * ENCODE ONCE, REUSE EVERYWHERE. The ~3400-candidate pool embedding + index +
    cross-encoder pass is the only expensive step (~24 min encode). We run the
    full scorer ONCE (CE on) and CACHE each scored candidate's feature vector,
    CE score, behavioural multiplier and honeypot flag. Weights touch ONLY
    ``combine()``, so every sweep config re-ranks from the cache in milliseconds
    — re-encoding per config is impossible by construction (the sweep never calls
    the encoder or the scorer again; it only re-runs the pure ``combine`` + floor).

  * TRAIN / VALIDATION SPLIT — the overfitting guard. The GRADED SILVER candidates
    (anchors EXCLUDED) are split deterministically (``config.SEED``), stratified by
    grade, into a TRAIN half and a held-out VALIDATION half. Weights are selected
    on the TRAIN composite ONLY; we then report the VALIDATION composite. A config
    that wins on train but not validation is overfitting — we flag it and prefer
    weights that generalise. The number we trust is the VALIDATION composite.

  * ANCHORS STAY SACRED. The manual anchors are NEVER in the tuning objective
    (neither train nor validation grade maps, and they are dropped from the
    train/val rankings so they neither reward nor penalise a config). They are
    used only for a final drift check.

  * COARSE, PRINCIPLED GRID. We sweep the three big levers — ``ce_score``,
    ``role_substance``, ``semantic_sim`` — and scale the eight small structured
    features together to fill the remainder so weights always sum to 1.0. A few
    dozen configs, not thousands; robustness over a fragile peak.

Eval-only. Imports ``scorer`` / ``metrics`` / ``evaluate`` READ-ONLY — weights and
``combine_fn`` pass through, so NOTHING in ``src/caliber`` is modified. This module
must never be imported by ``rank.py`` or any online module.

Run:
    # STEP A+B+C — encode once, persist ranking, print top 20, then STOP:
    python eval/sweep.py --step a --pool-size 3000
    # STEP D — sweep from the cache (no re-encode), after grades are finalised:
    python eval/sweep.py --step d
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from caliber import config  # noqa: E402
from caliber.scorer import (  # noqa: E402  (READ-ONLY: weights pass through)
    COMPOSITE_FEATURE_NAMES,
    DEFAULT_WEIGHTS,
    HONEYPOT_FLOOR,
    combine,
)
from eval.evaluate import (  # noqa: E402
    DEFAULT_MANUAL_GRADES,
    DEFAULT_SILVER_LABELS,
    evaluate_realistic,
    load_grades,
    load_manual_grades,
    resolve_grades,
)
from eval.metrics import evaluate_ranking  # noqa: E402

# The 9 structured-feature names = the composite vector minus the two
# description-level signals that the big levers control directly.
FEATURE_KEYS: tuple[str, ...] = tuple(
    n for n in COMPOSITE_FEATURE_NAMES if n not in ("ce_score", "semantic_sim")
)
# The small structured features scaled together (everything but role_substance).
_OTHER_FEATURE_KEYS: tuple[str, ...] = tuple(k for k in FEATURE_KEYS if k != "role_substance")

# Default artifact locations (artifacts/ is gitignored — derived state, never committed).
DEFAULT_CACHE_PATH = config.ARTIFACTS_DIR / "sweep" / "pool_cache.json"
DEFAULT_RANKING_CSV = config.ARTIFACTS_DIR / "sweep" / "ranking_default.csv"
DEFAULT_SWEEP_RESULTS = config.ARTIFACTS_DIR / "sweep" / "sweep_results.json"

# --- the coarse grid (constraint #4) --------------------------------------- #
CE_GRID = [0.0, 0.10, 0.20, 0.30, 0.40]
ROLE_GRID = [0.15, 0.23, 0.30, 0.40]
SEM_GRID = [0.10, 0.20, 0.30]


# --------------------------------------------------------------------------- #
# The cache: everything needed to recompute final_score for ANY weights.
# --------------------------------------------------------------------------- #
def cache_from_results(
    results: Mapping[str, Any], ranked_ids: Sequence[str]
) -> dict[str, Any]:
    """Project the scorer's ``{id: CandidateScore}`` into a JSON-able cache.

    Stores, per SCORED candidate, the exact inputs ``combine`` + the behavioural
    step + the honeypot floor consume — feature vector, CE score (None if the
    candidate fell past the CE head), semantic_sim, behavioural multiplier and the
    honeypot flag. ``all_pool_ids`` is the full pool (scored ∪ tail) so a re-rank
    can append the un-scored tail deterministically, exactly as ``rank_silver`` does.
    """
    scored: dict[str, Any] = {}
    for cid, cs in results.items():
        scored[str(cid)] = {
            "features": {k: cs.feature_dict.get(k) for k in FEATURE_KEYS},
            "ce_score": cs.ce_score,
            "semantic_sim": cs.semantic_sim,
            "behavioral_mult": cs.behavioral_mult,
            "is_honeypot": cs.is_honeypot,
            "default_final": cs.final_score,  # for the ranking CSV / provenance
        }
    return {"all_pool_ids": [str(c) for c in ranked_ids], "scored": scored}


def save_cache(cache: Mapping[str, Any], path: Any = DEFAULT_CACHE_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cache), encoding="utf-8")


def load_cache(path: Any = DEFAULT_CACHE_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Pure re-ranking from the cache (NO encode, NO scorer call).
# --------------------------------------------------------------------------- #
def recompute_final(entry: Mapping[str, Any], weights: Mapping[str, float]) -> float:
    """Reproduce ``score_candidates``' final score for one cached candidate under
    new ``weights``: ``combine`` over the (features + ce + semantic) vector, times
    the cached behavioural multiplier, then the honeypot floor LAST — identical to
    the scorer's steps 7-9, but driven entirely by cache."""
    if entry["is_honeypot"]:
        return HONEYPOT_FLOOR
    values: dict[str, Optional[float]] = dict(entry["features"])
    values["ce_score"] = entry["ce_score"]
    values["semantic_sim"] = entry["semantic_sim"]
    base = combine(values, weights)
    return base * entry["behavioral_mult"]


def rerank(cache: Mapping[str, Any], weights: Mapping[str, float]) -> list[str]:
    """Full ranking for ``weights``: scored candidates by ``final`` desc /
    candidate_id asc, then the un-scored tail sorted by candidate_id — exactly the
    deterministic contract ``rank_silver`` enforces."""
    scored = cache["scored"]
    finals = {cid: recompute_final(e, weights) for cid, e in scored.items()}
    ranked = sorted(scored.keys(), key=lambda cid: (-finals[cid], cid))
    tail = sorted(set(cache["all_pool_ids"]) - set(scored))
    return ranked + tail


# --------------------------------------------------------------------------- #
# Weight grid (renormalised to sum 1.0).
# --------------------------------------------------------------------------- #
def make_weights(ce: float, role: float, sem: float) -> Optional[dict[str, float]]:
    """A full 11-weight dict for the three lever values, with the eight small
    structured features scaled TOGETHER to fill the remainder so the dict sums to
    1.0. Returns ``None`` if the three levers already exceed 1.0 (infeasible — the
    small features cannot take negative weight)."""
    rem = 1.0 - ce - role - sem
    sum_others = sum(DEFAULT_WEIGHTS[k] for k in _OTHER_FEATURE_KEYS)
    if rem <= 1e-9 or sum_others <= 0.0:
        return None
    scale = rem / sum_others
    w = {k: DEFAULT_WEIGHTS[k] * scale for k in _OTHER_FEATURE_KEYS}
    w["role_substance"] = role
    w["ce_score"] = ce
    w["semantic_sim"] = sem
    # Guard the invariant the scorer relies on.
    assert abs(sum(w.values()) - 1.0) < 1e-9, sum(w.values())
    assert set(w) == set(COMPOSITE_FEATURE_NAMES)
    return w


def weight_grid() -> list[dict[str, float]]:
    """The feasible coarse grid (constraint #4); infeasible lever combos dropped."""
    grid: list[dict[str, float]] = []
    for ce in CE_GRID:
        for role in ROLE_GRID:
            for sem in SEM_GRID:
                w = make_weights(ce, role, sem)
                if w is not None:
                    grid.append(w)
    return grid


# --------------------------------------------------------------------------- #
# Train / validation split — stratified, deterministic, anchors excluded.
# --------------------------------------------------------------------------- #
def stratified_split(
    grades: Mapping[str, float], seed: int = config.SEED
) -> tuple[set[str], set[str]]:
    """Split graded ids into ``(train, val)`` — disjoint, stratified by integer
    grade, deterministic for a given ``seed``. Within each grade bucket the ids are
    sorted (stable) then shuffled by a per-bucket seeded RNG and halved, so both
    halves carry a similar grade distribution and two runs are identical."""
    by_grade: dict[int, list[str]] = defaultdict(list)
    for cid, g in grades.items():
        by_grade[int(g)].append(str(cid))
    train: set[str] = set()
    val: set[str] = set()
    for g in sorted(by_grade):
        ids = sorted(by_grade[g])
        random.Random(seed + g).shuffle(ids)
        half = len(ids) // 2
        train.update(ids[:half])
        val.update(ids[half:])
    return train, val


# --------------------------------------------------------------------------- #
# Metrics on a split (the other split + anchors removed from the ranking).
# --------------------------------------------------------------------------- #
def _subset_metrics(
    ranked_ids: Sequence[str],
    grades: Mapping[str, float],
    include_ids: set[str],
    exclude_ids: set[str],
    threshold: Optional[float] = None,
) -> dict[str, float]:
    """Composite over the ranking restricted to one split. ``exclude_ids`` (the
    OTHER split + anchors) are dropped from the ranking entirely so they neither
    reward nor penalise; ``include_ids`` are the graded positives; everything else
    (the unlabeled distractors) stays in as relevance 0 — the realistic sea."""
    ranked = [cid for cid in ranked_ids if cid not in exclude_ids]
    gmap = {cid: float(grades[cid]) for cid in include_ids if cid in grades}
    if threshold is None:
        return dict(evaluate_ranking(ranked, gmap))
    return dict(evaluate_ranking(ranked, gmap, threshold=threshold))


def _honeypots_in_top(cache: Mapping[str, Any], ranked_ids: Sequence[str], top: int) -> int:
    scored = cache["scored"]
    return sum(1 for cid in ranked_ids[:top] if scored.get(cid, {}).get("is_honeypot"))


# --------------------------------------------------------------------------- #
# STEP A + B — encode once, cache, persist the full ranking.
# --------------------------------------------------------------------------- #
def build_cache(
    *,
    pool_size: int = 3000,
    seed: int = config.SEED,
    ce_enabled: bool = True,
    silver_labels_path: Any = DEFAULT_SILVER_LABELS,
    manual_grades_path: Any = DEFAULT_MANUAL_GRADES,
    cache_path: Any = DEFAULT_CACHE_PATH,
    ranking_csv_path: Any = DEFAULT_RANKING_CSV,
    eval_fn: Optional[Callable] = None,
    **eval_kwargs: Any,
) -> dict[str, Any]:
    """Run the realistic scorer ONCE (CE on by default), cache the per-candidate
    scoring context, persist the full ranking, and return a summary incl. the top
    20. The expensive encode happens here and ONLY here; the sweep reuses the cache.
    """
    if eval_fn is None:
        eval_fn = evaluate_realistic

    out = eval_fn(
        pool_size=pool_size,
        seed=seed,
        ce_enabled=ce_enabled,
        silver_labels_path=silver_labels_path,
        manual_grades_path=manual_grades_path,
        **eval_kwargs,
    )

    cache = cache_from_results(out["results"], out["ranked_ids"])
    cache["meta"] = {
        "pool_size": pool_size,
        "seed": seed,
        "ce_enabled": ce_enabled,
        "n_scored": len(out["results"]),
        "encode_seconds": out.get("encode_seconds"),
        "total_seconds": out.get("total_seconds"),
    }

    # FAITHFULNESS self-check: re-ranking the cache with the DEFAULT weights must
    # reproduce the scorer's own ranking exactly — proves the cache captures the
    # full scoring state, so every sweep config is a valid recompute.
    check = rerank(cache, DEFAULT_WEIGHTS)
    if check != [str(c) for c in out["ranked_ids"]]:
        raise RuntimeError(
            "cache faithfulness check FAILED: re-ranking the cache with DEFAULT_WEIGHTS "
            "did not reproduce the scorer's ranking — the sweep would be invalid."
        )

    save_cache(cache, cache_path)

    # STEP B — persist the FULL ranking (rank, candidate_id, score, grade).
    grades_display, _ = resolve_grades(silver_labels_path, manual_grades_path)
    rows = _ranking_rows(out["ranked_ids"], out["results"], grades_display)
    _write_ranking_csv(rows, ranking_csv_path)

    return {
        "cache": cache,
        "cache_path": str(cache_path),
        "ranking_csv_path": str(ranking_csv_path),
        "ranked_ids": [str(c) for c in out["ranked_ids"]],
        "rows": rows,
        "encode_seconds": out.get("encode_seconds"),
        "total_seconds": out.get("total_seconds"),
        "report": out.get("report", ""),
    }


def _ranking_rows(
    ranked_ids: Sequence[str], results: Mapping[str, Any], grades: Mapping[str, float]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, cid in enumerate(ranked_ids, 1):
        cid = str(cid)
        cs = results.get(cid)
        score = getattr(cs, "final_score", None)
        g = grades.get(cid)
        rows.append({
            "rank": rank,
            "candidate_id": cid,
            "score": "" if score is None else round(float(score), 6),
            "grade": "?" if g is None else int(g),
        })
    return rows


def _write_ranking_csv(rows: Sequence[Mapping[str, Any]], path: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["rank", "candidate_id", "score", "grade"])
        w.writeheader()
        w.writerows(rows)


def format_top_n(rows: Sequence[Mapping[str, Any]], n: int = 20) -> str:
    lines = [f"Top {n} (default weights) — rank, candidate_id, score, grade:"]
    for r in rows[:n]:
        gtxt = f"grade={r['grade']}"
        lines.append(f"  {r['rank']:2d}. {r['candidate_id']}  score={r['score']:+.4f}  {gtxt}"
                     if isinstance(r["score"], float)
                     else f"  {r['rank']:2d}. {r['candidate_id']}  score=  {gtxt}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# STEP D — the sweep (no re-encode; pure recompute from the cache).
# --------------------------------------------------------------------------- #
def run_sweep(
    cache: Any,
    *,
    seed: int = config.SEED,
    silver_labels_path: Any = DEFAULT_SILVER_LABELS,
    manual_grades_path: Any = DEFAULT_MANUAL_GRADES,
    tuning_grades: Optional[Mapping[str, float]] = None,
    report_grades: Optional[Mapping[str, float]] = None,
    anchor_ids: Optional[set[str]] = None,
    grid: Optional[Sequence[Mapping[str, float]]] = None,
    top_k_report: int = 8,
) -> dict[str, Any]:
    """Sweep ``grid`` over the cached pool, select on TRAIN composite, report
    VALIDATION composite. Returns a structured result dict + a printable report.

    ``tuning_grades`` (silver, anchors EXCLUDED) drives the train/val split and the
    selection objective. ``report_grades`` (silver + anchors) and ``anchor_ids`` are
    used only for the final anchor-drift check — never for selection.
    """
    if isinstance(cache, (str, Path)):
        cache = load_cache(cache)

    # The tuning objective is silver grades ONLY — anchors are sacred (constraint #3).
    if tuning_grades is None:
        tuning_grades = load_grades(silver_labels_path)
    if anchor_ids is None:
        anchor_ids = set(load_manual_grades(manual_grades_path))
    if report_grades is None:
        report_grades, _ = resolve_grades(silver_labels_path, manual_grades_path)
    # Defensive: a tuning grade that is also an anchor must not be tuned against.
    tuning_grades = {cid: g for cid, g in tuning_grades.items() if cid not in anchor_ids}

    train, val = stratified_split(tuning_grades, seed)
    if grid is None:
        grid = weight_grid()

    train_excl = val | anchor_ids
    val_excl = train | anchor_ids

    configs: list[dict[str, Any]] = []
    for w in grid:
        ranked = rerank(cache, w)
        tr = _subset_metrics(ranked, tuning_grades, train, train_excl)
        configs.append({
            "weights": dict(w),
            "levers": {"ce_score": w["ce_score"], "role_substance": w["role_substance"],
                       "semantic_sim": w["semantic_sim"]},
            "train": tr,
            "ranked": ranked,
        })

    # Select on TRAIN composite (descending); break ties deterministically by levers.
    configs.sort(key=lambda c: (-c["train"]["composite"],
                                c["levers"]["ce_score"], c["levers"]["role_substance"],
                                c["levers"]["semantic_sim"]))

    # Compute VALIDATION metrics for the top-K-by-train (the generalisation check).
    head = configs[:top_k_report]
    for c in head:
        c["val"] = _subset_metrics(c["ranked"], tuning_grades, val, val_excl)

    # Baseline (current DEFAULT_WEIGHTS) on both splits, for side-by-side.
    base_ranked = rerank(cache, DEFAULT_WEIGHTS)
    baseline = {
        "weights": dict(DEFAULT_WEIGHTS),
        "levers": {k: DEFAULT_WEIGHTS[k] for k in ("ce_score", "role_substance", "semantic_sim")},
        "train": _subset_metrics(base_ranked, tuning_grades, train, train_excl),
        "val": _subset_metrics(base_ranked, tuning_grades, val, val_excl),
        "ranked": base_ranked,
    }

    # Recommend: best VALIDATION composite AMONG the top-train head (so we only
    # adopt a config that also led on train — guards against a val fluke).
    best = max(head, key=lambda c: c["val"]["composite"])

    # Honeypot + anchor drift checks for the recommended config (sanity, not objective).
    best_ranked = best["ranked"]
    honeypots = {f"top{t}": _honeypots_in_top(cache, best_ranked, t) for t in (10, 50, 100)}
    anchors = []
    pos = {cid: i + 1 for i, cid in enumerate(best_ranked)}
    for aid in sorted(anchor_ids):
        anchors.append({"candidate_id": aid, "rank": pos.get(aid),
                        "grade": report_grades.get(aid)})

    result = {
        "n_grid": len(grid),
        "train_size": len(train),
        "val_size": len(val),
        "n_anchors_excluded": len(anchor_ids),
        "baseline": baseline,
        "top_by_train": head,
        "recommended": best,
        "honeypots_top": honeypots,
        "anchors": anchors,
    }
    result["report"] = build_sweep_report(result)
    return result


def _lever_str(levers: Mapping[str, float]) -> str:
    return (f"ce={levers['ce_score']:.2f} role={levers['role_substance']:.2f} "
            f"sem={levers['semantic_sim']:.2f}")


def build_sweep_report(result: Mapping[str, Any]) -> str:
    L: list[str] = []
    L.append("=" * 78)
    L.append("CALIBER — WEIGHT SWEEP (select on TRAIN, trust VALIDATION)")
    L.append("=" * 78)
    L.append(f"  grid configs (feasible):   {result['n_grid']}")
    L.append(f"  train / val silver split:  {result['train_size']} / {result['val_size']}"
             f"   (anchors excluded: {result['n_anchors_excluded']})")
    L.append("")
    L.append("Top configs by TRAIN composite — with held-out VALIDATION (the honest number):")
    L.append(f"  {'levers':<34} {'train':>8} {'val':>8} {'val NDCG@10':>12} {'gap':>7}")
    for c in result["top_by_train"]:
        gap = c["train"]["composite"] - c["val"]["composite"]
        L.append(f"  {_lever_str(c['levers']):<34} "
                 f"{c['train']['composite']:>8.4f} {c['val']['composite']:>8.4f} "
                 f"{c['val']['ndcg@10']:>12.4f} {gap:>+7.4f}")
    L.append("  (gap = train - val; large positive gap => the config overfits the train half)")
    L.append("")

    base, best = result["baseline"], result["recommended"]
    L.append("BASELINE (current DEFAULT_WEIGHTS) vs BEST-FOUND — on VALIDATION:")
    L.append(f"  {'':<10} {'levers':<34} {'NDCG@10':>8} {'NDCG@50':>8} {'MAP':>7} {'P@10':>6} {'COMP':>8}")
    for tag, c in (("baseline", base), ("best", best)):
        v = c["val"]
        L.append(f"  {tag:<10} {_lever_str(c['levers']):<34} "
                 f"{v['ndcg@10']:>8.4f} {v['ndcg@50']:>8.4f} {v['map']:>7.4f} "
                 f"{v['p@10']:>6.3f} {v['composite']:>8.4f}")
    delta = best["val"]["composite"] - base["val"]["composite"]
    L.append(f"  validation composite delta (best - baseline): {delta:+.4f}")
    L.append("")
    L.append("Sanity (recommended config — NOT part of the objective):")
    hp = result["honeypots_top"]
    L.append(f"  honeypots in top 10 / 50 / 100:  {hp['top10']} / {hp['top50']} / {hp['top100']}")
    for a in result["anchors"]:
        L.append(f"  anchor {a['candidate_id']} (grade {a['grade']}): rank {a['rank']}")
    L.append("")
    L.append("Recommended weights (NOT auto-adopted — you decide):")
    for k in COMPOSITE_FEATURE_NAMES:
        L.append(f"    {k:<22} {best['weights'][k]:.4f}")
    L.append("=" * 78)
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Offline weight sweep over the realistic pool.")
    parser.add_argument("--step", choices=["a", "d"], required=True,
                        help="a = encode once + persist ranking + print top 20 (then STOP); "
                             "d = run the sweep from the cache (no re-encode).")
    parser.add_argument("--pool-size", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--cache", default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--ranking-csv", default=str(DEFAULT_RANKING_CSV))
    parser.add_argument("--top", type=int, default=20, help="how many ranks to print in step a")
    args = parser.parse_args(argv)

    if args.step == "a":
        print(f"[sweep] STEP A — encoding the realistic pool ONCE "
              f"(pool_size={args.pool_size}, seed={args.seed}, CE on)…")
        out = build_cache(pool_size=args.pool_size, seed=args.seed, ce_enabled=True,
                          cache_path=args.cache, ranking_csv_path=args.ranking_csv)
        enc = out["encode_seconds"]
        tot = out["total_seconds"]
        print(f"[sweep] encode time: {enc:.1f}s   total (encode+score+CE): {tot:.1f}s")
        print(f"[sweep] cache persisted     -> {out['cache_path']}")
        print(f"[sweep] full ranking (CSV)  -> {out['ranking_csv_path']}  "
              f"({len(out['rows'])} rows)")
        print("")
        print(format_top_n(out["rows"], args.top))
        print("")
        print(">>> STEP C HARD STOP: grade any '?' candidates in the top 20 by profile, "
              "add them to eval/manual_grades.json, then re-run with --step d.")
        return 0

    # step d
    print(f"[sweep] STEP D — sweeping from cache {args.cache} (no re-encode)…")
    result = run_sweep(args.cache, seed=args.seed)
    _save_sweep_results(result, DEFAULT_SWEEP_RESULTS)
    print(result["report"])
    print(f"[sweep] full sweep results -> {DEFAULT_SWEEP_RESULTS}")
    return 0


def _save_sweep_results(result: Mapping[str, Any], path: Any) -> None:
    """Persist the sweep result without the bulky per-config rankings."""
    def _slim(c: Mapping[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in c.items() if k != "ranked"}
    slim = {
        **{k: v for k, v in result.items() if k not in ("baseline", "top_by_train", "recommended")},
        "baseline": _slim(result["baseline"]),
        "top_by_train": [_slim(c) for c in result["top_by_train"]],
        "recommended": _slim(result["recommended"]),
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(slim, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

    
