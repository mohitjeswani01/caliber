"""Learning-to-rank model: train (OFFLINE) + predict (ONLINE).

Optional LightGBM (LambdaMART) ranker that learns to combine the structured
features + semantic/lexical scores into a single relevance score, trained
offline against our silver labels (see ``scripts/train_ltr.py``).

Responsibilities:
- ONLINE: load the persisted booster from ``artifacts/`` and predict a score per
  candidate from the assembled feature matrix. Inference is milliseconds — fits
  the budget comfortably.
- The trained model is a static artifact; no training happens in ``rank.py``.

LTR is an alternative/complement to hand-tuned linear fusion; whichever scores
higher on held-out silver labels wins. Determinism: fixed seed, single thread.
"""
