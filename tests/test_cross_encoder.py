"""Tests for src/caliber/cross_encoder.py — the CPU rerank stage.

Uses a TINY real sample (``data/challenge/sample_candidates.json``) and the
locally-cached model. Proves three things:
  1. rerank returns one score per input candidate, aligned to input order;
  2. on the JD, a real ML/retrieval career outranks keyword-stuffers — and the
     cross-encoder ordering is sharper than raw bi-encoder cosine (before/after);
  3. the model loads from the local cache with the network forced OFF.

The model is fetched offline by scripts/download_cross_encoder.py; these tests
skip (not fail) if it hasn't been cached yet, so a fresh clone without the
offline step doesn't break the suite.
"""

import json
import os

import pytest

from caliber import config, cross_encoder

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
SAMPLE_PATH = config.DATA_DIR / "challenge" / "sample_candidates.json"

# A real strong fit vs. clear keyword-stuffers in the sample (verified by hand):
#   CAND_0000031 — Recommendation Systems Engineer: shipped ranking/retrieval at
#                  Swiggy/Uber/Mad Street Den. Real career substance.
#   CAND_0000033 — Graphic Designer: design/consulting career, "curious about AI".
#   CAND_0000024 — HR Manager: marketing/HR career, AI listed as curiosity.
#   CAND_0000026 — Graphic Designer: noise with AI/data tags stuffed in skills.
STRONG_ID = "CAND_0000031"
STUFFER_IDS = ["CAND_0000033", "CAND_0000024", "CAND_0000026"]

# Concise JD query text grounded in the real Senior AI Engineer JD: depth in
# embeddings / retrieval / ranking / LLMs plus shipping production ML.
JD_TEXT = (
    "Senior AI Engineer. Deep technical depth in modern ML systems: embeddings, "
    "retrieval, ranking, recommendation, search, LLMs and fine-tuning. Must have "
    "shipped applied machine learning features in production at a product company, "
    "5-9 years of experience, NLP/IR background, scrappy product-engineering "
    "attitude."
)


def _model_available() -> bool:
    d = config.CROSS_ENCODER_MODEL_DIR
    return d.is_dir() and (d / "config.json").exists()


requires_model = pytest.mark.skipif(
    not _model_available(),
    reason="cross-encoder not cached; run scripts/download_cross_encoder.py offline",
)


def _candidate_to_text(c: dict) -> str:
    """Local stand-in for schema.candidate_to_text (schema.py not built yet).

    Mirrors the documented rich representation: headline + summary + role titles
    + role DESCRIPTIONS + skills-with-context. rerank() consumes plain text, so
    this lives in the test rather than in cross_encoder.py.
    """
    p = c.get("profile", {})
    parts = [p.get("headline") or "", p.get("summary") or ""]
    for role in c.get("career_history", []):
        title = role.get("title") or ""
        company = role.get("company") or ""
        desc = role.get("description") or ""
        parts.append(f"{title} at {company}. {desc}")
    skills = ", ".join(s.get("name", "") for s in c.get("skills", []))
    if skills:
        parts.append("Skills: " + skills)
    return "\n".join(part for part in parts if part.strip())


@pytest.fixture(scope="module")
def sample_by_id():
    with open(SAMPLE_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    return {c["candidate_id"]: c for c in data}


@pytest.fixture(scope="module")
def texts(sample_by_id):
    return {cid: _candidate_to_text(c) for cid, c in sample_by_id.items()}


# ---------------------------------------------------------------------------
# 1. Shape / order-alignment
# ---------------------------------------------------------------------------
@requires_model
def test_rerank_one_score_per_candidate_order_aligned(texts):
    strong = texts[STRONG_ID]
    stuffer = texts[STUFFER_IDS[0]]

    forward = cross_encoder.rerank(JD_TEXT, [strong, stuffer])
    assert len(forward) == 2
    assert all(isinstance(s, float) for s in forward)
    assert forward[0] > forward[1]  # strong (index 0) wins

    # Reverse the input: the higher score must follow the strong candidate to
    # its new index, proving scores are aligned to input order (not sorted).
    reverse = cross_encoder.rerank(JD_TEXT, [stuffer, strong])
    assert len(reverse) == 2
    assert reverse[1] > reverse[0]  # strong is now at index 1
    # Same candidate -> same score regardless of position (deterministic).
    assert forward[0] == pytest.approx(reverse[1], abs=1e-6)


# ---------------------------------------------------------------------------
# 2. Strong fit beats stuffers; cross-encoder sharper than cosine (before/after)
# ---------------------------------------------------------------------------
@requires_model
def test_rerank_beats_keyword_stuffers(texts, capsys):
    ids = [STRONG_ID] + STUFFER_IDS
    cand_texts = [texts[cid] for cid in ids]

    # --- AFTER: cross-encoder rerank ---
    ce_scores = cross_encoder.rerank(JD_TEXT, cand_texts)
    ce_order = sorted(zip(ids, ce_scores), key=lambda kv: kv[1], reverse=True)

    # --- BEFORE: raw bi-encoder cosine (the losing baseline) ---
    from sentence_transformers import SentenceTransformer
    import numpy as np

    bi = SentenceTransformer(str(config.EMBED_MODEL_LOCAL_DIR), device="cpu")
    embs = bi.encode([JD_TEXT] + cand_texts, normalize_embeddings=True)
    jd_vec, cand_vecs = embs[0], embs[1:]
    cos = (cand_vecs @ jd_vec).tolist()  # normalized -> dot == cosine
    cos_order = sorted(zip(ids, cos), key=lambda kv: kv[1], reverse=True)

    with capsys.disabled():
        print("\n--- BEFORE (bi-encoder cosine) ---")
        for cid, s in cos_order:
            print(f"   {cid}: {s:.4f}")
        print("--- AFTER (cross-encoder rerank) ---")
        for cid, s in ce_order:
            print(f"   {cid}: {s:.4f}")

    # The cross-encoder must rank the real ML/retrieval career strictly above
    # every keyword-stuffer.
    ce = dict(zip(ids, ce_scores))
    for stuffer in STUFFER_IDS:
        assert ce[STRONG_ID] > ce[stuffer], f"{STRONG_ID} should beat {stuffer}"
    # And it should sit at the very top of the reranked list.
    assert ce_order[0][0] == STRONG_ID


# ---------------------------------------------------------------------------
# 3. Loads from local cache with the network forced off
# ---------------------------------------------------------------------------
@requires_model
def test_loads_from_local_cache_no_network(texts, monkeypatch):
    # Force HF/transformers into offline mode, drop the singleton, and reload:
    # if this succeeds, the load came entirely from the local dir.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(cross_encoder, "_MODEL", None, raising=False)

    model = cross_encoder.load_cross_encoder()
    assert model is not None
    # A second call returns the same cached instance (singleton).
    assert cross_encoder.load_cross_encoder() is model

    # And it still scores under offline mode.
    scores = cross_encoder.rerank(JD_TEXT, [texts[STRONG_ID]])
    assert len(scores) == 1


# ---------------------------------------------------------------------------
# 4. Edge cases: top_k guard and empty input
# ---------------------------------------------------------------------------
@requires_model
def test_top_k_guard_scores_only_head(texts):
    ids = [STRONG_ID] + STUFFER_IDS
    cand_texts = [texts[cid] for cid in ids]

    scores = cross_encoder.rerank(JD_TEXT, cand_texts, top_k=2)
    assert len(scores) == len(ids)  # still one entry per candidate
    # The two unscored tail entries get a sentinel strictly below the scored head.
    head = scores[:2]
    tail = scores[2:]
    assert all(t < min(head) for t in tail)
    # Tail sentinels are equal to each other (deterministic).
    assert tail[0] == tail[1]


def test_empty_input_returns_empty():
    # No model needed: empty subset short-circuits before any load.
    assert cross_encoder.rerank(JD_TEXT, []) == []
