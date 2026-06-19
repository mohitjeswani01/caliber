"""OFFLINE training of the learning-to-rank model (LightGBM LambdaMART).

Trains a ``lambdarank`` booster to learn the base-relevance COMBINE step that
``scorer.combine`` does today by hand — over the SAME ordered
``scorer.COMPOSITE_FEATURE_NAMES`` feature vector — and reports, on a HELD-OUT
validation split, whether the learned model beats the tuned hand-weights
(``DEFAULT_WEIGHTS``). We adopt LTR ONLY if it clearly wins on validation; either
way we report both honestly.

Why this is skew-free and not overfit (the two things that matter):

  * NO FEATURE SKEW. We do NOT re-encode or re-extract features. We read the SAME
    per-candidate feature vectors the weight sweep already cached
    (``eval/sweep.py`` → ``artifacts/sweep/pool_cache.json``): the exact
    ``{role_substance, …, ce_score, semantic_sim}`` vectors ``scorer`` produces.
    Each training row is built with ``ltr.composite_vector`` — the identical
    ordering / NaN convention online inference uses. Train-time and score-time
    features are the same object by construction.

  * SAME ANTI-OVERFIT DISCIPLINE AS THE HAND-TUNING. We reuse
    ``sweep.stratified_split`` (``config.SEED``) to split the GRADED SILVER
    candidates (anchors EXCLUDED) into a TRAIN half and a held-out VALIDATION half.
    The model trains on the TRAIN half (+ the unlabeled distractors as relevance 0)
    ONLY; the VALIDATION positives and the manual anchors are NEVER in the training
    matrix. Hyper-parameters are fixed a-priori (regularised, not tuned on the val
    grades), so the validation composite is an honest held-out number — directly
    comparable to the sweep's baseline, computed with the SAME
    ``sweep._subset_metrics`` over the SAME split.

The booster touches ONLY the base combine: ``rerank_with_model`` still applies the
honeypot floor and the behavioural multiplier exactly as ``scorer`` /
``sweep.rerank`` do (LTR replaces the combine step, nothing else).

Deterministic: fixed seed, single thread, ``deterministic=True``. The trained
booster is saved to ``models/ltr.txt`` (gitignored) for online inference, with a
``models/ltr_meta.json`` provenance sidecar (features, params, the head-to-head).

Run:
    # 1) build the feature cache once (the expensive ~24-min encode), if absent:
    python eval/sweep.py --step a --pool-size 3000
    # 2) train + evaluate LTR vs the hand-weights on the held-out validation split:
    python scripts/train_ltr.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from caliber import config, ltr  # noqa: E402
from caliber.scorer import COMPOSITE_FEATURE_NAMES, DEFAULT_WEIGHTS, HONEYPOT_FLOOR  # noqa: E402
from eval import sweep  # noqa: E402  (reuse cache + split + metrics — no skew)
from eval.evaluate import (  # noqa: E402
    DEFAULT_MANUAL_GRADES,
    DEFAULT_SILVER_LABELS,
    load_manual_grades,
    resolve_grades,
)

# Default artifact locations. The model lives under MODELS_DIR (gitignored) next
# to the bge + cross-encoder weights; the cache is the sweep's pool cache.
DEFAULT_MODEL_OUT = ltr.LTR_MODEL_PATH                       # models/ltr.txt
DEFAULT_META_OUT = config.MODELS_DIR / "ltr_meta.json"
DEFAULT_CACHE_PATH = sweep.DEFAULT_CACHE_PATH               # artifacts/sweep/pool_cache.json

# Fixed, regularised hyper-parameters — chosen a-priori (small label set, ~150
# graded positives), NOT tuned on the validation grades, so the held-out number
# stays honest. Shallow trees + strong leaf regularisation guard against
# memorising the tiny silver set; lambdarank optimises the NDCG-shaped objective
# that dominates our composite.
DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [10, 50],
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 1.0,
    "lambda_l1": 0.0,
    "lambda_l2": 1.0,
    "max_depth": 4,
    "lambdarank_truncation_level": 50,   # pairs that matter for NDCG@10/@50
    "min_gain_to_split": 0.0,
    "verbosity": -1,
    # Determinism pins (ARCHITECTURE.md §5): one thread, fixed seed, no row-order
    # nondeterminism. With these the saved booster is bit-reproducible.
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 1,
    "seed": config.SEED,
}
DEFAULT_NUM_BOOST_ROUND = 300


# --------------------------------------------------------------------------- #
# Cache → composite values (identical assembly to sweep.recompute_final).
# --------------------------------------------------------------------------- #
def cache_values(entry: Mapping[str, Any]) -> dict[str, Optional[float]]:
    """The composite value dict for one cached candidate: the 9 structured
    features plus ``ce_score`` and ``semantic_sim`` — exactly the mapping
    ``scorer`` hands to its ``combine_fn`` (and what ``sweep.recompute_final``
    rebuilds). ``ltr.composite_vector`` then orders it into the model row."""
    values: dict[str, Optional[float]] = dict(entry["features"])
    values["ce_score"] = entry["ce_score"]
    values["semantic_sim"] = entry["semantic_sim"]
    return values


# --------------------------------------------------------------------------- #
# Build the training matrix (TRAIN positives + distractors=0; val & anchors out).
# --------------------------------------------------------------------------- #
def build_training_data(
    cache: Mapping[str, Any],
    graded: Mapping[str, float],
    train_ids: set[str],
    val_ids: set[str],
    anchor_ids: set[str],
    *,
    include_distractors: bool = True,
):
    """Assemble ``(X, y, used_ids)`` for LambdaMART from the cached pool.

    Each row is a cached candidate's composite vector (``ltr.composite_vector``)
    with a relevance label:
      - ``cid in train_ids``  → its silver grade (the supervised positives/zeros),
      - ungraded distractor   → relevance 0 (the realistic ~99%-irrelevant sea),
        included iff ``include_distractors`` (default: yes — they teach the model
        to push irrelevant profiles down, mirroring the real task),
      - ``cid in val_ids`` or ``cid in anchor_ids`` → SKIPPED (held out, never
        trained on — the overfitting guard).

    Rows are emitted in a deterministic order (candidate_id ascending) so the
    LightGBM dataset — and thus the saved booster — is reproducible. Returns
    ``(X: np.ndarray (n, n_features), y: list[float], used_ids: list[str])``.
    """
    scored = cache["scored"]
    rows: list[Mapping[str, Optional[float]]] = []
    y: list[float] = []
    used_ids: list[str] = []
    for cid in sorted(scored):
        if cid in val_ids or cid in anchor_ids:
            continue  # held out — never seen in training
        if cid in train_ids:
            label = float(graded[cid])
        elif cid in graded:
            # Graded but neither train nor val (shouldn't happen: split covers all
            # graded). Skip defensively rather than leak a val-adjacent label.
            continue
        else:
            if not include_distractors:
                continue
            label = 0.0
        rows.append(cache_values(scored[cid]))
        y.append(label)
        used_ids.append(cid)

    X = ltr.composite_matrix(rows)
    return X, y, used_ids


# --------------------------------------------------------------------------- #
# Train the LambdaMART booster (single query group = the whole pool vs the JD).
# --------------------------------------------------------------------------- #
def train_ranker(
    X,
    y: Sequence[float],
    *,
    params: Optional[Mapping[str, Any]] = None,
    num_boost_round: int = DEFAULT_NUM_BOOST_ROUND,
):
    """Train a LightGBM ``lambdarank`` booster on ``(X, y)`` as a SINGLE query group
    (every candidate is judged against the one JD), and return the ``Booster``.

    A single group is correct here: ranking quality is measured by one ordering of
    the whole pool against the JD, so the lambdas are computed over all
    candidate-pairs in that one ranking. Deterministic via ``DEFAULT_PARAMS``
    (seed, single thread). Imports lightgbm lazily so importing this module needs
    no LightGBM (the predict-side fallback and the rest of the suite stay usable).
    """
    import lightgbm as lgb

    p = dict(DEFAULT_PARAMS if params is None else params)
    n = len(y)
    if n == 0:
        raise ValueError("no training rows — empty cache or everything held out.")
    dataset = lgb.Dataset(X, label=list(y), group=[n], free_raw_data=False)
    booster = lgb.train(p, dataset, num_boost_round=num_boost_round)
    return booster


# --------------------------------------------------------------------------- #
# Rank the cached pool with the trained model — floor + behavioural PRESERVED.
# --------------------------------------------------------------------------- #
def model_final(entry: Mapping[str, Any], model: Any) -> float:
    """Reproduce ``score_candidates``' final score for one cached candidate, but
    with the LTR booster as the base combiner: honeypot → floor (LAST), else
    ``ltr.predict(values, model) × behavioural_mult``. Identical control flow to
    ``sweep.recompute_final`` — only the base combine differs, exactly as the spec
    requires (LTR replaces the combine step, NOT the floor / multiplier)."""
    if entry["is_honeypot"]:
        return HONEYPOT_FLOOR
    base = ltr.predict(cache_values(entry), model=model)
    return base * entry["behavioral_mult"]


def rerank_with_model(cache: Mapping[str, Any], model: Any) -> list[str]:
    """Full ranking under the LTR model: scored candidates by final desc /
    candidate_id asc, then the un-scored tail by candidate_id — the same
    deterministic contract ``sweep.rerank`` enforces for the hand-weights."""
    scored = cache["scored"]
    finals = {cid: model_final(e, model) for cid, e in scored.items()}
    ranked = sorted(scored.keys(), key=lambda cid: (-finals[cid], cid))
    tail = sorted(set(cache["all_pool_ids"]) - set(scored))
    return ranked + tail


# --------------------------------------------------------------------------- #
# Head-to-head: LTR vs hand-weights on TRAIN and the held-out VALIDATION split.
# --------------------------------------------------------------------------- #
def evaluate_head_to_head(
    cache: Mapping[str, Any],
    model: Any,
    tuning_grades: Mapping[str, float],
    train_ids: set[str],
    val_ids: set[str],
    anchor_ids: set[str],
) -> dict[str, Any]:
    """Compute TRAIN + VALIDATION composites for BOTH the LTR model and the tuned
    hand-weights, over the SAME ranking restrictions the sweep uses (the other
    split + anchors dropped). Returns a structured result for the report."""
    train_excl = val_ids | anchor_ids
    val_excl = train_ids | anchor_ids

    ltr_ranked = rerank_with_model(cache, model)
    base_ranked = sweep.rerank(cache, DEFAULT_WEIGHTS)

    def metrics(ranked: Sequence[str]) -> dict[str, dict[str, float]]:
        return {
            "train": sweep._subset_metrics(ranked, tuning_grades, train_ids, train_excl),
            "val": sweep._subset_metrics(ranked, tuning_grades, val_ids, val_excl),
        }

    ltr_m = metrics(ltr_ranked)
    base_m = metrics(base_ranked)

    def hp(ranked: Sequence[str]) -> dict[str, int]:
        return {f"top{t}": sweep._honeypots_in_top(cache, ranked, t) for t in (10, 50, 100)}

    return {
        "ltr": ltr_m,
        "baseline": base_m,
        "ltr_ranked": ltr_ranked,
        "base_ranked": base_ranked,
        "val_delta": ltr_m["val"]["composite"] - base_m["val"]["composite"],
        "honeypots_ltr": hp(ltr_ranked),
        "honeypots_baseline": hp(base_ranked),
    }


def build_report(result: Mapping[str, Any], *, adopt_threshold: float, adopt: bool,
                 train_size: int, val_size: int, n_anchors: int,
                 n_train_rows: int, n_pos_rows: int) -> str:
    L: list[str] = []
    L.append("=" * 78)
    L.append("CALIBER — LEARNING-TO-RANK vs HAND-WEIGHTS (held-out validation)")
    L.append("=" * 78)
    L.append(f"  train / val silver split:  {train_size} / {val_size}"
             f"   (anchors held out: {n_anchors})")
    L.append(f"  training rows (incl. distractors=0):  {n_train_rows}"
             f"   of which graded positives/zeros: {n_pos_rows}")
    L.append("")
    L.append("Composite on each split (LTR base-combine vs tuned DEFAULT_WEIGHTS):")
    L.append(f"  {'':<10} {'NDCG@10':>8} {'NDCG@50':>8} {'MAP':>7} {'P@10':>6} {'COMP':>8}")
    for tag, m in (("hand TR", result["baseline"]["train"]),
                   ("LTR  TR", result["ltr"]["train"]),
                   ("hand VAL", result["baseline"]["val"]),
                   ("LTR  VAL", result["ltr"]["val"])):
        L.append(f"  {tag:<10} {m['ndcg@10']:>8.4f} {m['ndcg@50']:>8.4f} "
                 f"{m['map']:>7.4f} {m['p@10']:>6.3f} {m['composite']:>8.4f}")
    L.append("")
    delta = result["val_delta"]
    L.append(f"VALIDATION composite delta (LTR - hand-weights): {delta:+.4f}")
    L.append(f"adopt threshold (LTR must beat hand-weights by): {adopt_threshold:+.4f}")
    L.append("")
    hp_l, hp_b = result["honeypots_ltr"], result["honeypots_baseline"]
    L.append("Sanity (honeypots in top 10 / 50 / 100 — must stay 0 in the head):")
    L.append(f"  LTR:        {hp_l['top10']} / {hp_l['top50']} / {hp_l['top100']}")
    L.append(f"  hand-weights:{hp_b['top10']} / {hp_b['top50']} / {hp_b['top100']}")
    L.append("")
    if adopt:
        L.append(f"DECISION: ADOPT LTR — it beats the hand-weights by {delta:+.4f} on the")
        L.append("          held-out validation split. Wire scorer with combine_fn=ltr.predict.")
    else:
        L.append(f"DECISION: KEEP HAND-WEIGHTS — LTR's validation gain ({delta:+.4f}) does not")
        L.append(f"          clear the adopt threshold ({adopt_threshold:+.4f}). 'LTR ≈ hand-weights'")
        L.append("          is a valid, honest finding; the model artifact is still saved for audit.")
    L.append("=" * 78)
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Persistence.
# --------------------------------------------------------------------------- #
def save_model(model: Any, path: Any = DEFAULT_MODEL_OUT) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(p))


def save_meta(meta: Mapping[str, Any], path: Any = DEFAULT_META_OUT) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #
def run(
    *,
    cache_path: Any = DEFAULT_CACHE_PATH,
    silver_labels_path: Any = DEFAULT_SILVER_LABELS,
    manual_grades_path: Any = DEFAULT_MANUAL_GRADES,
    model_out: Any = DEFAULT_MODEL_OUT,
    meta_out: Any = DEFAULT_META_OUT,
    seed: int = config.SEED,
    num_boost_round: int = DEFAULT_NUM_BOOST_ROUND,
    include_distractors: bool = True,
    adopt_threshold: float = 0.0,
    params: Optional[Mapping[str, Any]] = None,
    build_cache: bool = False,
    pool_size: int = 3000,
) -> dict[str, Any]:
    """Train, evaluate head-to-head, persist, and return the result dict.

    ``adopt_threshold`` is how much LTR must beat the hand-weights by on VALIDATION
    composite to be recommended for adoption (default 0.0 → any strict win; raise
    it to demand a clear margin). Adoption is a RECOMMENDATION in the report — it
    never rewrites ``DEFAULT_WEIGHTS`` or ``scorer``; a human wires
    ``combine_fn=ltr.predict`` only if we accept it.
    """
    cache_path = Path(cache_path)
    if build_cache and not cache_path.exists():
        print(f"[train_ltr] cache absent — building it once (pool_size={pool_size}, CE on)…")
        sweep.build_cache(pool_size=pool_size, seed=seed, ce_enabled=True,
                          cache_path=cache_path)
    if not cache_path.exists():
        raise SystemExit(
            f"[train_ltr] feature cache not found at {cache_path}.\n"
            f"  Build it first:  python eval/sweep.py --step a --pool-size {pool_size}\n"
            f"  (or re-run this with --build-cache to build it inline)."
        )

    cache = sweep.load_cache(cache_path)

    # Grades: silver + manual anchors merged for reporting; anchors are EXCLUDED
    # from the tuning objective (sacred — never trained on, never split).
    report_grades, n_manual = resolve_grades(silver_labels_path, manual_grades_path)
    anchor_ids = set(load_manual_grades(manual_grades_path))
    tuning_grades = {cid: g for cid, g in sweep.load_grades(silver_labels_path).items()
                     if cid not in anchor_ids}

    train_ids, val_ids = sweep.stratified_split(tuning_grades, seed)

    X, y, used_ids = build_training_data(
        cache, tuning_grades, train_ids, val_ids, anchor_ids,
        include_distractors=include_distractors,
    )
    n_pos_rows = sum(1 for cid in used_ids if cid in train_ids)

    model = train_ranker(X, y, params=params, num_boost_round=num_boost_round)
    save_model(model, model_out)

    result = evaluate_head_to_head(
        cache, model, tuning_grades, train_ids, val_ids, anchor_ids
    )
    adopt = result["val_delta"] > adopt_threshold

    meta = {
        "seed": seed,
        "num_boost_round": num_boost_round,
        "params": dict(DEFAULT_PARAMS if params is None else params),
        "feature_names": list(COMPOSITE_FEATURE_NAMES),
        "include_distractors": include_distractors,
        "n_train_rows": len(y),
        "n_positive_train_rows": n_pos_rows,
        "train_size": len(train_ids),
        "val_size": len(val_ids),
        "n_anchors_excluded": len(anchor_ids),
        "val_composite_ltr": result["ltr"]["val"]["composite"],
        "val_composite_handweights": result["baseline"]["val"]["composite"],
        "val_delta": result["val_delta"],
        "adopt_threshold": adopt_threshold,
        "recommended_adopt": adopt,
        "model_path": str(Path(model_out)),
    }
    save_meta(meta, meta_out)

    report = build_report(
        result, adopt_threshold=adopt_threshold, adopt=adopt,
        train_size=len(train_ids), val_size=len(val_ids), n_anchors=len(anchor_ids),
        n_train_rows=len(y), n_pos_rows=n_pos_rows,
    )
    return {"result": result, "meta": meta, "report": report, "adopt": adopt,
            "model": model}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train + evaluate the LightGBM LTR ranker vs the hand-weights (OFFLINE)."
    )
    parser.add_argument("--cache", default=str(DEFAULT_CACHE_PATH),
                        help="sweep feature cache (artifacts/sweep/pool_cache.json).")
    parser.add_argument("--silver-labels", default=str(DEFAULT_SILVER_LABELS))
    parser.add_argument("--manual-grades", default=str(DEFAULT_MANUAL_GRADES))
    parser.add_argument("--model-out", default=str(DEFAULT_MODEL_OUT))
    parser.add_argument("--meta-out", default=str(DEFAULT_META_OUT))
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--num-boost-round", type=int, default=DEFAULT_NUM_BOOST_ROUND)
    parser.add_argument("--no-distractors", action="store_true",
                        help="train on graded silver only (exclude the unlabeled "
                             "distractors that are treated as relevance 0).")
    parser.add_argument("--adopt-threshold", type=float, default=0.0,
                        help="min VALIDATION composite margin (LTR - hand) to "
                             "recommend adopting LTR (default 0.0 = any strict win).")
    parser.add_argument("--build-cache", action="store_true",
                        help="build the feature cache inline if it is absent "
                             "(runs the expensive encode; otherwise it must exist).")
    parser.add_argument("--pool-size", type=int, default=3000,
                        help="pool size if --build-cache builds the cache.")
    args = parser.parse_args(argv)

    out = run(
        cache_path=args.cache,
        silver_labels_path=args.silver_labels,
        manual_grades_path=args.manual_grades,
        model_out=args.model_out,
        meta_out=args.meta_out,
        seed=args.seed,
        num_boost_round=args.num_boost_round,
        include_distractors=not args.no_distractors,
        adopt_threshold=args.adopt_threshold,
        build_cache=args.build_cache,
        pool_size=args.pool_size,
    )
    print(out["report"])
    print(f"[train_ltr] model saved   -> {args.model_out}")
    print(f"[train_ltr] meta  saved   -> {args.meta_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
