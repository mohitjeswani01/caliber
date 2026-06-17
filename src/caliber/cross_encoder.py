"""CPU cross-encoder re-rank of the shortlist (ONLINE, budgeted).

After cheap retrieval (bi-encoder cosine + BM25) and rank fusion narrow the
100K pool to a small shortlist (~800), this module re-scores each
``(JD, candidate-text)`` PAIR *jointly* with a local cross-encoder. Unlike the
bi-encoder — which embeds JD and candidate independently and compares with a
single dot product — the cross-encoder runs full cross-attention over the pair,
so it can tell real retrieval/ranking career substance apart from keyword
stuffing. That sharpens the head of the list (NDCG@10/@50), where most of the
metric lives.

Strictly budget-aware and offline-safe:
- Runs ONLY over the passed subset (the shortlist), NEVER the full 100K — a
  cross-encoder pays a forward pass per pair, so 100K pairs would blow the
  5-minute CPU budget.
- Model (``cross-encoder/ms-marco-MiniLM-L-6-v2``) is loaded from the local dir
  ``config.CROSS_ENCODER_MODEL_DIR``, populated offline by
  ``scripts/download_cross_encoder.py``. NO network call at rank time.
- Deterministic: CPU, eval mode, fixed seed, stable input order in → stable
  scores out.

This is a precision booster for the head of the list, not a crutch: substance
ranking must stand on its own if the rerank is ever budget-gated off.
"""

from __future__ import annotations

import os
import time
from typing import Iterable, Optional

from . import config

# Module-level singleton so the model loads from disk exactly once per process.
_MODEL = None


def load_cross_encoder():
    """Load the cross-encoder from the local cache (singleton, CPU, no network).

    Loads only from ``config.CROSS_ENCODER_MODEL_DIR``. The model must have been
    fetched offline first (``python scripts/download_cross_encoder.py``); if the
    dir is missing we fail loudly rather than silently reaching for the network.

    Mirrors ``embeddings.load_model``'s fail-closed contract: by default this
    hard-locks HF/transformers to offline mode *before* importing them, so even a
    *partial/corrupt* local dir (a half-finished download missing a weight or
    tokenizer file) errors out instead of silently reaching the HuggingFace hub —
    a hub fetch in the judged online path is a Stage-3 disqualification. The ONLY
    opt-in to allow a download is the offline phase setting
    ``CALIBER_ALLOW_MODEL_DOWNLOAD=1`` (same env var as the embedding model);
    everything else, above all the judged ``rank.py`` path, is offline-locked.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    model_dir = config.CROSS_ENCODER_MODEL_DIR
    # Explicit, opt-in escape hatch — ONLY the offline precompute/download phase
    # sets this. Everything else (above all the judged online path) is locked.
    allow_download = os.environ.get("CALIBER_ALLOW_MODEL_DOWNLOAD") == "1"

    if not allow_download:
        # Defense in depth: lock HF/transformers to offline BEFORE importing them
        # below, so the online path can never reach the network even by accident.
        # A partial/corrupt local dir then errors out instead of silently hitting
        # the hub to "repair" itself.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if not (model_dir.is_dir() and (model_dir / "config.json").exists()):
            raise FileNotFoundError(
                f"Cross-encoder not found at {model_dir}. Run "
                f"`python scripts/download_cross_encoder.py` once offline to cache it. "
                f"rank.py must never download at runtime."
            )

    # Imported lazily so merely importing this module (e.g. when testing
    # fusion.py) doesn't drag in torch/sentence-transformers.
    import torch
    from sentence_transformers import CrossEncoder

    torch.manual_seed(config.SEED)  # belt-and-suspenders; inference is deterministic
    # max_length MUST be set: candidate texts exceed the model's 512-token limit,
    # and an unset cap both warns of indexing errors and pays attention cost on
    # the wasted tail. Truncating to a fixed length also keeps timing predictable.
    model = CrossEncoder(
        str(model_dir),
        device="cpu",
        max_length=config.CROSS_ENCODER_MAX_LENGTH,
    )
    model.model.eval()
    _MODEL = model
    return _MODEL


def _to_text(item) -> str:
    """Coerce a shortlist element to candidate text.

    Accepts either a pre-built text string (what ``schema.candidate_to_text``
    produces — the orchestrator builds these for the shortlist) or a candidate
    object, which is converted via ``schema.candidate_to_text``. Keeping the
    text-building in ``schema`` means this module never reinvents it.
    """
    if isinstance(item, str):
        return item
    from .schema import candidate_to_text  # lazy: schema is owned elsewhere
    return candidate_to_text(item)


def rerank(
    jd_text: str,
    candidates_subset: Iterable,
    top_k: Optional[int] = None,
) -> list[float]:
    """Cross-encoder score every ``(jd_text, candidate_text)`` pair in the subset.

    Args:
        jd_text: the JD query text (one string, scored against each candidate).
        candidates_subset: ordered shortlist. Each element is either candidate
            text (``str``) or a candidate object convertible via
            ``schema.candidate_to_text``. This is the ~800-item shortlist — NEVER
            pass the full pool.
        top_k: optional cost guard. If given, only the first ``top_k`` candidates
            (the subset is assumed best-first from the cheap stage) are scored by
            the model; the remainder receive a sentinel strictly below every
            scored value so they sort after the reranked head. ``None`` scores
            the whole subset.

    Returns:
        A list of floats, one per input candidate, **aligned to input order**
        (higher = better fit). Empty input → empty list.
    """
    subset = list(candidates_subset)
    n = len(subset)
    if n == 0:
        return []

    model = load_cross_encoder()

    n_score = n if top_k is None else max(0, min(top_k, n))
    pairs = [[jd_text, _to_text(subset[i])] for i in range(n_score)]

    t0 = time.perf_counter()
    if pairs:
        raw = model.predict(
            pairs,
            batch_size=config.CROSS_ENCODER_BATCH_SIZE,
            show_progress_bar=False,
        )
        scored = [float(x) for x in raw]
    else:
        scored = []
    elapsed = time.perf_counter() - t0

    per_pair = (elapsed / len(pairs) * 1000.0) if pairs else 0.0
    print(
        f"[cross_encoder.rerank] scored {len(pairs)} pairs (of {n} in subset) "
        f"in {elapsed:.3f}s ({per_pair:.1f} ms/pair) on CPU"
    )

    if n_score == n:
        return scored

    # Tail beyond top_k was not reranked: give a finite sentinel below every
    # scored value so it sorts strictly after the reranked head, deterministically.
    sentinel = (min(scored) - 1.0) if scored else 0.0
    return scored + [sentinel] * (n - n_score)
