"""Local sentence-transformer encoding (OFFLINE) + query encoding (ONLINE).

Owns the single embedding model used in both phases (``BAAI/bge-small-en-v1.5``,
384-dim), loaded from a locally-cached path so neither phase hits the network.

Responsibilities:
- Build the rich per-candidate text representation (delegated to
  ``schema.candidate_to_text`` — headline + summary + role titles + role
  **descriptions** + skills-with-context) and encode all 100K candidates to
  ``candidate_emb.npy`` (offline, batched, memory-safe).
- Encode the JD query text(s) online (a handful of vectors — trivially cheap).

Encoding the *descriptions*, not the skill tags, is what surfaces hidden gems.

The model is downloaded once during offline precompute and saved under
``models/`` (``config.EMBED_MODEL_LOCAL_DIR``); after that ``load_model`` resolves
it from local disk, so the online ranker makes zero network calls.
"""

from __future__ import annotations

import os

from typing import Any, Iterable, List, Mapping, Union

import numpy as np

from . import config
from .schema import candidate_to_text

# Module-level singleton so repeated calls (precompute + online query encoding)
# reuse one loaded model instead of paying the load cost twice.
_MODEL = None

# A candidate may be passed either as a raw record dict or as an already-built
# text string (lets callers pre-build / cache text if they want to).
CandidateLike = Union[str, Mapping[str, Any]]


def load_model():
    """Load the embedding model (CPU) from the LOCAL cache. Fails closed.

    The online ranker (``rank.py``) makes ZERO network calls — a single hit to
    the HuggingFace hub in the judged sandbox is a Stage-3 disqualification. So
    this loader never downloads by default: if the model is not already cached at
    ``config.EMBED_MODEL_LOCAL_DIR`` it raises instead of silently fetching it,
    and it hard-locks HF/transformers to offline mode *before* importing them.

    The ONE legitimate place a download is allowed is the offline precompute
    phase, which opts in by setting the env var ``CALIBER_ALLOW_MODEL_DOWNLOAD=1``
    (e.g. ``CALIBER_ALLOW_MODEL_DOWNLOAD=1 python scripts/precompute.py`` on a
    fresh box). With that flag set, the offline guards are NOT applied and a
    missing model is fetched from the hub and cached to the local dir for every
    subsequent (offline-locked) load. Once cached, neither path touches the
    network again.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    local_dir = config.EMBED_MODEL_LOCAL_DIR
    # Explicit, opt-in escape hatch — ONLY the offline precompute phase sets this.
    # Everything else (above all the judged online path) is offline-locked.
    allow_download = os.environ.get("CALIBER_ALLOW_MODEL_DOWNLOAD") == "1"

    if not allow_download:
        # Defense in depth: lock HF/transformers to offline BEFORE importing them
        # below, so the online path can never reach the network even by accident —
        # a stray fetch errors out instead of silently hitting the hub.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if not local_dir.exists():
            raise RuntimeError(
                f"Embedding model not found at {local_dir}. The online/rank path "
                "makes ZERO network calls, so the model must already be cached "
                "locally — this loader will not download it. Run the offline "
                "precompute once to download and cache the model first:\n"
                "    CALIBER_ALLOW_MODEL_DOWNLOAD=1 python scripts/precompute.py"
            )

    # Imported lazily so merely importing this module (e.g. in tests that only
    # touch candidate_to_text) does not drag in torch / sentence-transformers.
    from sentence_transformers import SentenceTransformer

    if local_dir.exists():
        model = SentenceTransformer(str(local_dir), device="cpu")
    else:
        # Reachable only with CALIBER_ALLOW_MODEL_DOWNLOAD=1 (offline precompute,
        # first run): fetch from the hub and cache locally for every later load.
        model = SentenceTransformer(config.EMBED_MODEL_NAME, device="cpu")
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model.save(str(local_dir))

    # Cap sequence length (see config): bounds compute and keeps encoding
    # deterministic regardless of how long an individual profile's text is.
    model.max_seq_length = config.EMBED_MAX_SEQ_LENGTH

    _MODEL = model
    return model


def _encode(texts: List[str], batch_size: int) -> np.ndarray:
    """Encode a list of texts to float32, L2-normalized vectors."""
    model = load_model()
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2 normalize so inner product == cosine
        show_progress_bar=False,
    )
    return np.ascontiguousarray(emb, dtype=np.float32)


def encode_candidates(
    candidates: Iterable[CandidateLike], batch_size: int = 256
) -> np.ndarray:
    """Encode candidates to a ``(N, EMBED_DIM)`` float32, L2-normalized array.

    Row ``i`` corresponds to the ``i``-th candidate yielded by ``candidates`` —
    this ordering is the join key with ``candidate_ids.npy`` and the FAISS index,
    so the caller must iterate the pool in a stable order (file order).

    Streams: at most ``batch_size`` candidate texts are held in memory at once,
    so this stays well within the 16 GB budget (and the ~5 GB dev box) even on
    the full 100K pool. The accumulated embedding array is only ~154 MB.
    """
    chunks: List[np.ndarray] = []
    buffer: List[str] = []

    def flush() -> None:
        if buffer:
            chunks.append(_encode(buffer, batch_size))
            buffer.clear()

    for candidate in candidates:
        text = candidate if isinstance(candidate, str) else candidate_to_text(candidate)
        buffer.append(text)
        if len(buffer) >= batch_size:
            flush()
    flush()

    if not chunks:
        return np.empty((0, config.EMBED_DIM), dtype=np.float32)
    return np.vstack(chunks)


def encode_texts(
    texts: List[str], is_query: bool = False, batch_size: int = 64
) -> np.ndarray:
    """Encode a handful of texts (e.g. the JD aspect queries) — ONLINE-cheap.

    Set ``is_query=True`` for the JD/query side so the bge retrieval instruction
    is prepended (asymmetric s2p retrieval — candidates are the "passages" and
    get no prefix). Returns float32, L2-normalized vectors.
    """
    if is_query:
        texts = [config.BGE_QUERY_INSTRUCTION + t for t in texts]
    return _encode(texts, batch_size)
