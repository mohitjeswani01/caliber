"""Tests for the embedding pipeline: candidate_to_text, the encoder, and precompute row-order wiring."""

import importlib.util
import json

import numpy as np
import pytest

from caliber import config
from caliber.schema import candidate_to_text

SAMPLE = config.DATA_DIR / "challenge" / "sample_candidates.json"


def _load_sample(n=None):
    recs = json.loads(SAMPLE.read_text(encoding="utf-8"))
    return recs[:n] if n is not None else recs


# --- text builder (fast, no model) -----------------------------------------

def test_candidate_to_text_includes_role_descriptions():
    rec = _load_sample(1)[0]
    text = candidate_to_text(rec)
    # The whole point: role free-text descriptions are embedded, not just titles.
    first_desc = rec["career_history"][0]["description"]
    assert first_desc[:50] in text
    assert rec["profile"]["headline"] in text
    assert rec["profile"]["summary"][:50] in text


def test_candidate_to_text_tolerates_missing_fields():
    sparse = {"candidate_id": "CAND_0000000", "profile": {},
              "career_history": [], "skills": []}
    assert isinstance(candidate_to_text(sparse), str)
    assert candidate_to_text({}) == ""


# --- encoder (uses the cached model) ----------------------------------------

@pytest.fixture(scope="module")
def _model():
    from caliber import embeddings
    try:
        embeddings.load_model()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"embedding model unavailable: {exc}")
    return embeddings


def test_encode_candidates_shape_dtype_and_normalized(_model):
    emb = _model.encode_candidates(_load_sample(4), batch_size=2)
    assert emb.shape == (4, config.EMBED_DIM)
    assert emb.dtype == np.float32
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-4)


def test_encode_candidates_row_order_matches_input(_model):
    recs = _load_sample(3)
    batch = _model.encode_candidates(recs, batch_size=8)
    # Row i must equal candidate i encoded alone (order preserved + deterministic).
    for i, rec in enumerate(recs):
        solo = _model.encode_candidates([rec], batch_size=1)[0]
        assert np.allclose(batch[i], solo, atol=1e-5)


def test_encode_candidates_accepts_prebuilt_text(_model):
    rec = _load_sample(1)[0]
    from_dict = _model.encode_candidates([rec])[0]
    from_text = _model.encode_candidates([candidate_to_text(rec)])[0]
    assert np.allclose(from_dict, from_text, atol=1e-6)


def test_encode_texts_query_instruction_changes_vector(_model):
    as_query = _model.encode_texts(["retrieval ranking search systems"], is_query=True)
    as_passage = _model.encode_texts(["retrieval ranking search systems"], is_query=False)
    assert as_query.shape == (1, config.EMBED_DIM)
    # The bge query instruction is prepended only when is_query -> vectors differ.
    assert not np.allclose(as_query[0], as_passage[0])


# --- precompute wiring: row order is the join key (ids <-> emb) --------------

@pytest.fixture(scope="module")
def _precompute():
    spec = importlib.util.spec_from_file_location(
        "precompute", config.ROOT / "scripts" / "precompute.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_precompute_artifacts_align_ids_with_rows(_model, _precompute, tmp_path):
    if not config.CANDIDATES_PATH.exists():
        pytest.skip("data/candidates.jsonl not present")

    n = 6
    _precompute.build_embeddings_and_index(
        config.CANDIDATES_PATH, tmp_path, batch_size=4, limit=n, shard_size=4)

    emb = np.load(tmp_path / config.CANDIDATE_EMB_FILE)
    ids = np.load(tmp_path / config.CANDIDATE_IDS_FILE)
    assert emb.shape == (n, config.EMBED_DIM)
    assert emb.dtype == np.float32
    assert ids.shape[0] == emb.shape[0]

    # ids must equal the first n candidate_ids from the file, in order.
    expected = [json.loads(line)["candidate_id"]
                for _, line in zip(range(n), open(config.CANDIDATES_PATH))]
    assert list(ids) == expected

    # And the saved index round-trips against the saved embeddings.
    from caliber import index
    idx = index.load_index(tmp_path / config.FAISS_INDEX_FILE)
    assert idx.ntotal == n
    _, hit = index.search(idx, emb[0], k=1)
    assert int(hit[0, 0]) == 0
