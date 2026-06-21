"""Tests for rank.py — the CLI orchestrator that produces the submission CSV.

These run on a TINY synthetic pool with FAKE retrieval seams (a numpy stand-in for
the FAISS index, exactly like eval/evaluate.py and tests/test_scorer.py), so no
model, no faiss/torch, and no 100K pool are needed — yet they exercise the real
end-to-end path: load/inject artifacts → score → build rows → write CSV →
self-validate with tests/validate_submission.py.

Pinned here:
  * a clean pool produces a CSV that PASSES the official validator (header, exactly
    100 rows, ranks 1..100, non-increasing score, candidate_id tie-break, no dupes);
  * every row's reasoning column is populated;
  * two runs are byte-identical (determinism);
  * a missing-artifact dir raises ArtifactError (no silent 100K re-encode / network);
  * the honeypot guardrail propagates: a breaching pool raises and writes NO CSV.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Make src/ + repo root importable and load rank.py by path (it lives at repo root).
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_spec = importlib.util.spec_from_file_location("rank", ROOT / "rank.py")
rank = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rank)

from caliber.ranker import HoneypotGuardrailError  # noqa: E402
from caliber.schema import parse_candidate  # noqa: E402


# --------------------------------------------------------------------------- #
# JD profile (two weighted aspects → exercises aspect-weighted aggregation).
# --------------------------------------------------------------------------- #
JD = {
    "role": "Senior AI Engineer",
    "experience_band": {"min": 5, "max": 9, "ideal_min": 6, "ideal_max": 8},
    "consulting_firms": ["TCS", "Infosys", "Wipro", "Accenture"],
    "location_prefs": {"country_priority": "India"},
    "aspects": {
        "a_retrieval": {
            "weight": 0.6,
            "query_text": "embeddings semantic search dense retrieval in production",
            "keywords": ["embeddings", "semantic search", "dense retrieval"],
        },
        "b_ranking": {
            "weight": 0.4,
            "query_text": "learning to rank ranking relevance ndcg",
            "keywords": ["ranking", "learning to rank", "ndcg"],
        },
    },
}


# --------------------------------------------------------------------------- #
# Candidate factory (full schema-valid records that survive parse_candidate()).
# --------------------------------------------------------------------------- #
def _signals(**over):
    base = dict(
        profile_completeness_score=0.9, signup_date="2020-01-01",
        last_active_date="2026-06-01", open_to_work_flag=True,
        profile_views_received_30d=10, applications_submitted_30d=2,
        recruiter_response_rate=0.6, avg_response_time_hours=12.0,
        skill_assessment_scores={}, connection_count=300, endorsements_received=50,
        notice_period_days=30, expected_salary_range_inr_lpa={"min": 20.0, "max": 40.0},
        preferred_work_mode="hybrid", willing_to_relocate=True,
        github_activity_score=40.0, search_appearance_30d=5, saved_by_recruiters_30d=3,
        interview_completion_rate=0.9, offer_acceptance_rate=0.5,
        verified_email=True, verified_phone=True, linkedin_connected=True,
    )
    base.update(over)
    return base


def _role(title, months, desc, current=True, start="2022-06-01", end=None):
    return dict(company="Acme", title=title, start_date=start,
                end_date=end if not current else None, duration_months=months,
                is_current=current, industry="Software", company_size="501-1000",
                description=desc)


def _cand(cid, title, yoe, roles, skills, *, location="Bangalore", country="India",
          summary="", signals=None):
    return dict(
        candidate_id=cid,
        profile=dict(anonymized_name="T", headline=title, summary=summary,
                     location=location, country=country, years_of_experience=yoe,
                     current_title=title, current_company="Acme",
                     current_company_size="501-1000", current_industry="Software"),
        career_history=roles, education=[],
        skills=[dict(name=s, proficiency="advanced", endorsements=5, duration_months=24)
                for s in skills],
        certifications=[], languages=[], redrob_signals=signals or _signals(),
    )


def _strong(cid):
    return _cand(
        cid, "Senior AI Engineer", 7.0,
        [_role("Senior AI Engineer", 30,
               "Built and deployed embeddings-based semantic search and a "
               "learning-to-rank system in production; measured NDCG and MRR.")],
        ["NLP", "Information Retrieval", "Embeddings", "Ranking"],
        summary="NLP/IR engineer building retrieval and ranking systems in production.",
    )


def _filler(cid, *, yoe=5.0):
    # An adjacent-but-thin profile: scores positive, well below the strong fits, so
    # the ordering is unambiguous and there are plenty to fill 100 rows.
    return _cand(
        cid, "Software Engineer", yoe,
        [_role("Software Engineer", 36, "Built backend services and data pipelines.")],
        ["Python", "SQL"], summary="Backend engineer.",
    )


def _honeypot(cid):
    # Internally impossible: a single role longer than the whole career.
    return _cand(
        cid, "Senior AI Engineer", 3.0,
        [_role("Senior AI Engineer", 120,
               "Built embeddings retrieval and learning-to-rank systems; NDCG.")],
        ["Embeddings", "Ranking", "NLP"],
        signals=_signals(github_activity_score=90.0, last_active_date="2026-06-16",
                         recruiter_response_rate=1.0),
    )


# --------------------------------------------------------------------------- #
# Fake retrieval seams (no model, no faiss): an (N, D) matrix of unit vectors and
# a search_fn that reproduces IndexFlatIP (cosine = dot product), top-k desc.
# --------------------------------------------------------------------------- #
def _normalize(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _build_pool(records):
    """Return (candidates_by_id, candidate_ids, fake_index, encode_fn, search_fn).

    Strong candidates get a vector aligned with the JD aspect queries; everyone else
    gets a weak/orthogonal vector, so the strong fits rise to the top deterministically.
    """
    cands = {r["candidate_id"]: parse_candidate(r) for r in records}
    ids = sorted(cands)  # deterministic row order (== candidate_ids.npy contract)

    # 2-D toy embedding space: aspect queries point at [1, 0]; strong cands align.
    def cand_vec(cid):
        title = cands[cid].profile.current_title.lower()
        if "ai engineer" in title and cands[cid].profile.years_of_experience >= 5:
            return [1.0, 0.05]          # strong fits: close to the query direction
        return [0.1, 1.0]               # everyone else: mostly orthogonal

    emb = np.stack([_normalize(cand_vec(cid)) for cid in ids]).astype(np.float32)

    def encode_fn(texts, is_query=False):
        # Both aspect queries point the same way in this toy space.
        return np.stack([_normalize([1.0, 0.0]) for _ in texts]).astype(np.float32)

    def search_fn(index, query_emb, k):
        q = np.ascontiguousarray(query_emb, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        sims = q @ emb.T                      # (Q, N) cosine
        k = min(k, emb.shape[0])
        idx = np.argsort(-sims, axis=1)[:, :k]
        sc = np.take_along_axis(sims, idx, axis=1)
        return sc.astype(np.float32), idx.astype(np.int64)

    return cands, ids, emb, encode_fn, search_fn


def _run(records, tmp_path, *, top_n, out_path=None, **kw):
    cands, ids, emb, encode_fn, search_fn = _build_pool(records)
    return rank.produce_submission(
        out_path=out_path if out_path is not None else tmp_path / "submission.csv",
        top_n=top_n,
        ce_enabled=False,                      # no CE model in the test env
        jd_profile=JD,
        candidate_ids=ids,
        faiss_index=emb,                       # the fake search_fn ignores it
        candidates_by_id=cands,
        encode_query_fn=encode_fn,
        search_fn=search_fn,
        verbose=False,
        **kw,
    )


# --------------------------------------------------------------------------- #
# 1. Clean pool → valid CSV that PASSES the official validator.
# --------------------------------------------------------------------------- #
def test_end_to_end_produces_valid_csv(tmp_path):
    # 5 strong fits + enough filler to exceed 100 rows.
    records = [_strong(f"CAND_{i:07d}") for i in range(1, 6)]
    records += [_filler(f"CAND_{i:07d}", yoe=4.0 + (i % 5)) for i in range(100, 230)]

    out = _run(records, tmp_path, top_n=100)

    assert out["validation_errors"] == []          # passes validate_submission.py
    assert out["n_rows"] == 100
    assert out["honeypots_in_top"] == 0

    # Re-validate the file independently and inspect its shape.
    import csv
    with open(out["out_path"], encoding="utf-8", newline="") as fh:
        reader = list(csv.reader(fh))
    assert reader[0] == ["candidate_id", "rank", "score", "reasoning"]
    assert len(reader) == 101                       # header + 100
    ranks = [int(r[1]) for r in reader[1:]]
    assert ranks == list(range(1, 101))
    scores = [float(r[2]) for r in reader[1:]]
    assert all(a >= b for a, b in zip(scores, scores[1:]))   # non-increasing
    ids = [r[0] for r in reader[1:]]
    assert len(set(ids)) == 100                     # no dupes
    # The 5 strong fits are at the very top.
    assert set(ids[:5]) == {f"CAND_{i:07d}" for i in range(1, 6)}


def test_every_row_has_reasoning(tmp_path):
    records = [_strong(f"CAND_{i:07d}") for i in range(1, 6)]
    records += [_filler(f"CAND_{i:07d}") for i in range(100, 230)]
    out = _run(records, tmp_path, top_n=100)

    import csv
    with open(out["out_path"], encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 100
    assert all(row["reasoning"].strip() for row in rows)
    assert all("\n" not in row["reasoning"] for row in rows)
    # The strong fits read as such; threaded Candidate gives the title+years lead.
    top = rows[0]["reasoning"]
    assert top.startswith("Senior AI Engineer, 7 yrs")
    assert "Strong fit" in top


# --------------------------------------------------------------------------- #
# 2. Determinism — two runs byte-identical.
# --------------------------------------------------------------------------- #
def test_deterministic_byte_identical(tmp_path):
    records = [_strong(f"CAND_{i:07d}") for i in range(1, 6)]
    records += [_filler(f"CAND_{i:07d}") for i in range(100, 230)]

    out1 = _run(records, tmp_path / "a", top_n=100)
    out2 = _run(records, tmp_path / "b", top_n=100)
    a = Path(out1["out_path"]).read_bytes()
    b = Path(out2["out_path"]).read_bytes()
    assert a == b
    assert out1["validation_errors"] == [] == out2["validation_errors"]


# --------------------------------------------------------------------------- #
# 3. Missing-artifact path — clear error, no silent re-encode / network.
# --------------------------------------------------------------------------- #
def test_missing_artifacts_raise_clear_error(tmp_path):
    empty = tmp_path / "no_artifacts"
    empty.mkdir()
    with pytest.raises(rank.ArtifactError) as ei:
        rank.produce_submission(
            candidates_path=tmp_path / "unused.jsonl",
            artifacts_dir=empty,
            out_path=tmp_path / "out.csv",
            verbose=False,
        )
    msg = str(ei.value)
    assert "precompute" in msg          # tells the user how to fix it
    assert not (tmp_path / "out.csv").exists()   # nothing written


# --------------------------------------------------------------------------- #
# 4. Honeypot guardrail propagates — breaching pool raises, NO CSV written.
# --------------------------------------------------------------------------- #
def test_honeypot_guardrail_aborts_without_writing(tmp_path):
    # top_n=10 with 2 strong honeypots among 8 reals → 20% ≥ 10% limit. The
    # honeypots are floored to -1.0 by the scorer, so to actually breach the top-10
    # we DISABLE the floor's natural exclusion by making the pool exactly 10 — the
    # floored honeypots still occupy 2 of the 10 selected slots.
    records = [_honeypot("CAND_0000901"), _honeypot("CAND_0000902")]
    records += [_filler(f"CAND_{i:07d}") for i in range(1, 9)]   # 8 reals → pool of 10
    out_csv = tmp_path / "should_not_exist.csv"

    with pytest.raises(HoneypotGuardrailError):
        _run(records, tmp_path, top_n=10, out_path=out_csv)
    assert not out_csv.exists()         # guardrail fires BEFORE any write


def test_floored_honeypots_excluded_when_pool_is_large(tmp_path):
    # With enough reals, floored honeypots (-1.0) sort below the cut and never enter
    # the top-100 at all — the guardrail count is 0.
    records = [_strong(f"CAND_{i:07d}") for i in range(1, 6)]
    records += [_filler(f"CAND_{i:07d}") for i in range(100, 230)]
    records += [_honeypot("CAND_0000901"), _honeypot("CAND_0000902")]
    out = _run(records, tmp_path, top_n=100)
    assert out["honeypots_in_top"] == 0
    assert out["validation_errors"] == []
