"""FAISS index build (OFFLINE) and semantic search (ONLINE).

Persists a FAISS index over the candidate embeddings to ``artifacts/`` and
provides fast cosine/IP similarity lookups online.

Responsibilities:
- Build the index from ``candidate_emb.npy``. The row order of the embeddings is
  the join key back to ``candidate_ids.npy`` — index position ``i`` is always
  candidate ``i`` — so results map to candidate_ids deterministically.
- Online: given the JD query vector(s), return per-candidate semantic similarity
  scores within the CPU budget.

Pure retrieval. Note: semantic similarity is ONE input to the hybrid score —
never the sole ranking signal (raw cosine is the losing baseline).

We use ``IndexFlatIP`` on L2-normalized vectors, so inner product == cosine
similarity. Flat (exact) search over 100K x 384 is well within the CPU/5-min
budget and avoids the recall loss / nondeterminism of approximate indexes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import faiss


def build_index(emb: np.ndarray) -> faiss.Index:
    """Build an exact inner-product index over (already L2-normalized) vectors."""
    emb = np.ascontiguousarray(emb, dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError(f"expected a 2-D (N, dim) array, got shape {emb.shape}")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index


def save_index(index: faiss.Index, path: Union[str, Path]) -> None:
    """Persist a FAISS index to disk (creates parent dirs as needed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: Union[str, Path]) -> faiss.Index:
    """Load a FAISS index previously written by :func:`save_index`."""
    return faiss.read_index(str(path))


def search(
    index: faiss.Index, query_emb: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(scores, indices)`` for the top-``k`` neighbours per query.

    ``query_emb`` may be a single vector ``(dim,)`` or a batch ``(Q, dim)``;
    it is reshaped/cast to contiguous float32. ``indices`` are row positions into
    the embedding matrix — map them to candidate_ids via ``candidate_ids.npy``.
    """
    q = np.ascontiguousarray(query_emb, dtype=np.float32)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    return index.search(q, k)
