"""OFFLINE training of the learning-to-rank model.

Trains the LightGBM LambdaMART ranker (``src/caliber/ltr.py``) on the assembled
feature matrix against the silver labels, optimising a ranking objective aligned
with the composite metric (NDCG-weighted). Persists the booster to
``artifacts/`` for online inference.

Fixed seed and single-threaded determinism so the artifact is reproducible.
Reports held-out NDCG@10/@50/MAP via ``eval/metrics.py`` so we keep LTR only if
it beats hand-tuned fusion. Stub only — no logic yet.
"""

if __name__ == "__main__":
    raise SystemExit("train_ltr.py is a stub — not implemented yet.")
