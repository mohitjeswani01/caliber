"""Tests for the FAISS index module (build / save / load / search).

Pure vector tests — no embedding model needed, so these run fast and offline.
"""

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from caliber import config, index


def _unit_vectors(seed: int, n: int, d: int = config.EMBED_DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, d)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def test_build_index_dim_and_count():
    v = _unit_vectors(0, 20)
    idx = index.build_index(v)
    assert idx.ntotal == 20
    assert idx.d == config.EMBED_DIM


def test_build_index_rejects_non_2d():
    with pytest.raises(ValueError):
        index.build_index(np.zeros(config.EMBED_DIM, dtype=np.float32))


def test_round_trip_returns_planted_neighbor(tmp_path):
    v = _unit_vectors(42, 50)
    idx = index.build_index(v)
    path = tmp_path / "faiss.index"
    index.save_index(idx, path)
    loaded = index.load_index(path)

    # A query identical to candidate row 7 must come back as the top-1 hit with
    # cosine ~= 1.0 (inner product on unit vectors).
    scores, ids = index.search(loaded, v[7], k=3)
    assert ids.shape == (1, 3)
    assert scores.shape == (1, 3)
    assert int(ids[0, 0]) == 7
    assert abs(float(scores[0, 0]) - 1.0) < 1e-4


def test_search_accepts_1d_and_2d_queries():
    v = _unit_vectors(1, 10)
    idx = index.build_index(v)
    s1, i1 = index.search(idx, v[0], k=2)        # 1-D single query
    s2, i2 = index.search(idx, v[0:1], k=2)      # 2-D (1, dim) query
    assert i1.shape == (1, 2) and i2.shape == (1, 2)
    assert int(i1[0, 0]) == 0 and int(i2[0, 0]) == 0
    assert np.allclose(s1, s2)
