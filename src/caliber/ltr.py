"""Learning-to-rank model: predict (ONLINE) + graceful fallback.

An optional LightGBM (LambdaMART / ``lambdarank``) ranker that learns to combine
the SAME ordered ``scorer.COMPOSITE_FEATURE_NAMES`` feature vector into a single
base-relevance score — replacing ONLY the hand-weighted ``scorer.combine`` step.
It is trained OFFLINE against our silver labels (``scripts/train_ltr.py``) and
persisted to a local artifact; the online path here does inference only.

How it plugs in (no change to scorer's feature assembly):

    from caliber import ltr, scorer
    scorer.score_candidates(..., combine_fn=ltr.predict)

``predict`` has the exact signature ``scorer`` calls its combiner with —
``combine_fn(values, weights)`` — so it is a drop-in for ``scorer.combine``.
The honeypot floor, behavioural multiplier and feature extraction all stay in
``scorer``; this module only maps the assembled feature vector → a number.

Graceful fallback (NON-NEGOTIABLE — must never break ``rank.py``):
- If LightGBM is not importable, OR the trained model artifact is absent, OR the
  booster fails to load, ``predict`` transparently falls back to the hand-weighted
  ``scorer.combine`` over the SAME values/weights. So setting
  ``combine_fn=ltr.predict`` is always safe: with a model it uses the model, and
  without one it is byte-identical to the current hand-weighted scorer.

Determinism: inference is a pure function of the feature vector; single-threaded
``Booster.predict``. CPU-only, no network — the booster is read from a local file
(``models/ltr.txt``); we never download anything at score time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from . import config
from .scorer import COMPOSITE_FEATURE_NAMES, DEFAULT_WEIGHTS, _sigmoid, combine

# --------------------------------------------------------------------------- #
# Artifact location. The LightGBM booster is saved in its native text format
# (``Booster.save_model``) under MODELS_DIR, alongside the bge + cross-encoder
# weights (ARCHITECTURE.md §3/§4). Gitignored like every other model file.
# --------------------------------------------------------------------------- #
LTR_MODEL_FILE = "ltr.txt"
LTR_MODEL_PATH = config.MODELS_DIR / LTR_MODEL_FILE

# Process-wide load cache so ``predict`` (called once per shortlisted candidate)
# loads the booster from disk at most once. Keyed by resolved path so a test that
# points at a different artifact is not served a stale model. ``model is None``
# is a CACHED outcome too — a missing/ uninstalled model is not re-probed 800×.
_MODEL_CACHE: dict[str, Optional[Any]] = {}


# --------------------------------------------------------------------------- #
# Feature vectorization — the ONLY place the value dict becomes a model row, so
# train-time and score-time agree by construction (train_ltr.py reuses this).
# --------------------------------------------------------------------------- #
def composite_vector(values: Mapping[str, Optional[float]]) -> np.ndarray:
    """Order ``values`` into the fixed ``COMPOSITE_FEATURE_NAMES`` row the model
    expects. A missing / ``None`` feature (e.g. ``ce_score`` when the cross-encoder
    did not reach this candidate) becomes ``np.nan`` — LightGBM handles NaN natively
    by learning a default split direction, mirroring how ``combine`` simply drops a
    ``None`` weight. Returns a float64 vector of length ``len(COMPOSITE_FEATURE_NAMES)``.
    """
    row = np.empty(len(COMPOSITE_FEATURE_NAMES), dtype=np.float64)
    for i, name in enumerate(COMPOSITE_FEATURE_NAMES):
        v = values.get(name)
        row[i] = np.nan if v is None else float(v)
    return row


def composite_matrix(rows: Sequence[Mapping[str, Optional[float]]]) -> np.ndarray:
    """Stack ``composite_vector`` over many value dicts → an ``(n, n_features)``
    float64 matrix. Used by ``train_ltr.py`` so the training matrix is built with
    the EXACT same feature ordering / NaN convention as online inference."""
    if not rows:
        return np.empty((0, len(COMPOSITE_FEATURE_NAMES)), dtype=np.float64)
    return np.vstack([composite_vector(r) for r in rows])


# --------------------------------------------------------------------------- #
# Model loading (cached, fail-closed).
# --------------------------------------------------------------------------- #
def load_model(path: Optional[Any] = None, *, force: bool = False) -> Optional[Any]:
    """Load and cache the LightGBM booster from ``path`` (default ``LTR_MODEL_PATH``).

    Returns the ``Booster`` on success, or ``None`` when the model is unavailable —
    LightGBM not installed, the artifact file is absent, or it fails to parse. A
    ``None`` result is cached so the (common, expected) no-model case does not
    re-probe the filesystem / re-attempt the import on every candidate. Pass
    ``force=True`` to bypass the cache (tests that write a fresh artifact).

    Never raises: a missing model is a normal, documented state (we then fall back
    to the hand-weights), not an error.
    """
    if path is None:
        path = LTR_MODEL_PATH
    key = str(Path(path))
    if not force and key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    model: Optional[Any] = None
    p = Path(path)
    if not p.exists():
        _MODEL_CACHE[key] = None
        return None
    try:
        import lightgbm as lgb  # offline-only dependency; import lazily

        model = lgb.Booster(model_file=str(p))
    except ImportError:
        print("[ltr] lightgbm not installed — falling back to hand-weighted combine.")
        model = None
    except Exception as exc:  # corrupt / incompatible artifact: degrade, never crash
        print(f"[ltr] failed to load model at {p}: {exc!r} — falling back to hand-weights.")
        model = None

    _MODEL_CACHE[key] = model
    return model


def clear_cache() -> None:
    """Drop the cached booster(s). Tests call this after writing a new artifact so
    the next ``load_model`` re-reads from disk."""
    _MODEL_CACHE.clear()


def is_available(path: Optional[Any] = None) -> bool:
    """True iff a usable trained model is present (so callers / reports can state
    plainly whether the LTR path or the hand-weight fallback is active)."""
    return load_model(path) is not None


# --------------------------------------------------------------------------- #
# Predict — the drop-in ``combine_fn``.
# --------------------------------------------------------------------------- #
def predict(
    values: Mapping[str, Optional[float]],
    weights: Optional[Mapping[str, float]] = None,
    *,
    model: Optional[Any] = None,
    model_path: Optional[Any] = None,
) -> float:
    """Base-relevance score for ONE candidate's assembled feature vector.

    Signature matches ``scorer.combine(values, weights)`` so this is a drop-in
    ``combine_fn``: ``scorer`` calls it as ``predict(composite_values, weights)``.

    - With a trained model present, ``weights`` is ignored and the booster scores
      the ordered feature vector. The booster's RAW lambdarank output is unbounded,
      so we squash it through ``scorer._sigmoid`` → base ∈ (0, 1). This is required
      to stay a faithful drop-in for ``combine`` (also (0,1)): it keeps the honeypot
      FLOOR (-1.0, strictly below any real score) valid and the behavioural
      multiplier operating on a [0,1] base. The sigmoid is monotonic, so the learned
      RANKING is unchanged — only the scale is normalised.
    - With NO model (LightGBM missing / artifact absent), this falls back to
      ``scorer.combine`` over the same ``values`` and ``weights`` — so the scorer
      behaves EXACTLY as it does today with the hand-weights. ``rank.py`` is never
      broken by a missing artifact.

    ``model`` / ``model_path`` are injection seams for ``train_ltr.py`` and tests
    (score against an in-memory booster, or a non-default artifact) without
    touching the process-wide cache.
    """
    if model is None:
        model = load_model(model_path)
    if model is None:
        # Fail closed: hand-weighted combine over the identical vector.
        return combine(values, weights if weights is not None else DEFAULT_WEIGHTS)

    row = composite_vector(values).reshape(1, -1)
    pred = model.predict(row, num_threads=1)
    return _sigmoid(float(pred[0]))  # squash unbounded score → base ∈ (0,1)


def predict_batch(
    rows: Sequence[Mapping[str, Optional[float]]],
    weights: Optional[Mapping[str, float]] = None,
    *,
    model: Optional[Any] = None,
    model_path: Optional[Any] = None,
) -> list[float]:
    """Vectorized ``predict`` over many candidates (one ``Booster.predict`` call).

    Same fallback semantics as ``predict`` — with no model, returns the
    hand-weighted ``combine`` for each row. Handy for offline evaluation where we
    score a whole pool at once; the online scorer uses the per-candidate ``predict``
    through its ``combine_fn`` slot.
    """
    if model is None:
        model = load_model(model_path)
    if model is None:
        w = weights if weights is not None else DEFAULT_WEIGHTS
        return [combine(r, w) for r in rows]
    if not rows:
        return []
    preds = model.predict(composite_matrix(rows), num_threads=1)
    return [_sigmoid(float(x)) for x in preds]  # squash → base ∈ (0,1), as predict
