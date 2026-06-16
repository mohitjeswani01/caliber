"""OFFLINE silver-label generation — our own ground truth (our edge).

We are given NO labels and NO leaderboard (STRATEGY.md §7), so we build our own
relevance answer key and *measure* before we spend one of our 3 submissions.
This is the thin ORCHESTRATOR: it wires together the focused ``eval/`` modules,
runs the pipeline, and writes the outputs. It is **eval-only** — it must NEVER be
imported by ``rank.py`` or any ``src/caliber`` online module (that would be
training the ranker on its own test set). Outputs live under ``eval/``.

Pipeline (all offline, no time/compute budget, ZERO API calls from code):

  STEP 1  eval/sampling.py  — stratified, deterministic sample (~400).
  STEP 2  eval/rubric.py    — rule-based 0-4 grader + transparent breakdown.
  STEP 3  (here)            — write batched LLM grading prompts; read grades back.
                              We do NOT call any API from code. The human runs
                              eval/llm_grading_prompts.jsonl through Claude offline
                              and returns eval/llm_grades.jsonl. If absent, the
                              script still runs (rule-only fallback).
  STEP 4  eval/agreement.py — rules-vs-LLM agreement + reconciliation.
  STEP 5  eval/anchors.py   — sacred unambiguous anchor set + anchor check.

Run:  python scripts/make_silver_labels.py
      python scripts/make_silver_labels.py --candidates ./data/candidates.jsonl
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

# Make the repo root importable so ``eval`` and ``src`` resolve regardless of CWD.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.agreement import agreement_report, reconcile  # noqa: E402
from eval.anchors import build_anchors, check_anchors  # noqa: E402
from eval.heuristics import _profile, _roles, _signals, _skills  # noqa: E402
from eval.rubric import grade_rules  # noqa: E402
from eval.sampling import (  # noqa: E402
    ALL_STRATA,
    STRATUM_TARGETS,
    TOTAL_TARGET,
    scan_and_bucket,
    select_sample,
)

# config is pure constants (not part of the ONLINE ranking path).
try:
    from src.caliber.config import REFERENCE_DATE, SEED
except Exception:  # pragma: no cover - fallback if config is unavailable
    SEED = 42
    REFERENCE_DATE = "2026-06-16"

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
DEFAULT_CANDIDATES = ROOT / "data" / "candidates.jsonl"
DEFAULT_JD_PROFILE = ROOT / "artifacts" / "jd_profile.json"
EVAL_DIR = ROOT / "eval"

OUT_LABELS = EVAL_DIR / "silver_labels.json"
OUT_SAMPLE_IDS = EVAL_DIR / "silver_sample_ids.json"
OUT_LLM_PROMPTS = EVAL_DIR / "llm_grading_prompts.jsonl"
OUT_ANCHORS = EVAL_DIR / "anchors.json"
IN_LLM_GRADES = EVAL_DIR / "llm_grades.jsonl"


# --------------------------------------------------------------------------- #
# STEP 3 — LLM grading round-trip (NO API calls from code).
# We emit a prompt per candidate; the human runs them through Claude offline and
# returns eval/llm_grades.jsonl as {candidate_id, grade_llm, reason} lines.
# --------------------------------------------------------------------------- #
RUBRIC_TEXT = (
    "You are grading how well a candidate fits a *Senior AI Engineer* role on a "
    "0-4 relevance scale. Judge CAREER SUBSTANCE, not keyword presence — a profile "
    "that merely lists 'RAG/LLM/NLP' is worth nothing unless the career history "
    "shows real retrieval, ranking, recommendation, search or applied-ML work at a "
    "product company.\n\n"
    "GRADES:\n"
    "  4 = strong fit: right title/substance, 6-8 yrs (5-9 ok), product-company "
    "experience, genuine NLP/IR + retrieval/ranking work, no disqualifiers.\n"
    "  3 = good fit: most must-haves present, only minor gaps.\n"
    "  2 = partial/adjacent: some relevant signal but notable gaps.\n"
    "  1 = weak/tangential.\n"
    "  0 = irrelevant, OR a honeypot (impossible internals), OR a keyword-stuffer "
    "(non-tech career stuffed with AI skills it never used).\n\n"
    "FORCE LOW (cap near 0-1): career entirely at consulting/services firms; "
    "primary computer-vision/speech/robotics with no NLP/IR; pure research with no "
    "production; only <12-month LangChain-calls-OpenAI with no prior ML; senior who "
    "hasn't shipped code in 18+ months; title-chaser (job-hops < ~1.5 yrs).\n"
)
JD_SUMMARY = (
    "ROLE: Senior AI Engineer. Wants 5-9 yrs (ideal 6-8) building embeddings-based "
    "retrieval, vector/hybrid search, and ranking/recommendation systems IN "
    "PRODUCTION, with rigorous ranking evaluation (NDCG/MRR/MAP) and a real NLP/IR "
    "background (not primarily CV/speech/robotics), at product companies (not "
    "career-long services/consulting). India Tier-1 or relocation-willing preferred."
)


def _compact_candidate(c):
    p = _profile(c)
    sig = _signals(c)
    lines = [
        f"candidate_id: {c.get('candidate_id')}",
        f"current_title: {p.get('current_title')} | years_of_experience: {p.get('years_of_experience')}",
        f"location: {p.get('location')}, {p.get('country')} | company: {p.get('current_company')} "
        f"({p.get('current_company_size')}, {p.get('current_industry')})",
        f"headline: {p.get('headline')}",
        f"summary: {(p.get('summary') or '')[:500]}",
        "career_history (most recent first):",
    ]
    for r in _roles(c):
        cur = ", current" if r.get("is_current") else ""
        lines.append(
            f"  - {r.get('title')} @ {r.get('company')} ({r.get('duration_months')}mo, "
            f"{r.get('start_date')}..{r.get('end_date')}{cur}) [{r.get('industry')}, {r.get('company_size')}]"
        )
        desc = (r.get("description") or "")[:320]
        if desc:
            lines.append(f"      {desc}")
    skills = _skills(c)[:15]
    sk = "; ".join(
        f"{s.get('name')}({s.get('proficiency')},{s.get('duration_months', '?')}mo)" for s in skills
    )
    lines.append(f"skills: {sk}")
    lines.append(
        "signals: "
        f"github_activity_score={sig.get('github_activity_score')}, "
        f"last_active_date={sig.get('last_active_date')}, "
        f"recruiter_response_rate={sig.get('recruiter_response_rate')}, "
        f"open_to_work={sig.get('open_to_work_flag')}, "
        f"interview_completion_rate={sig.get('interview_completion_rate')}, "
        f"notice_period_days={sig.get('notice_period_days')}, "
        f"willing_to_relocate={sig.get('willing_to_relocate')}"
    )
    return "\n".join(lines)


def build_llm_prompt(c, jd_summary=JD_SUMMARY):
    """A single self-contained grading prompt (rubric + JD + candidate + output
    contract). Deterministic for a given candidate."""
    return (
        f"{RUBRIC_TEXT}\n{jd_summary}\n\n"
        f"CANDIDATE:\n{_compact_candidate(c)}\n\n"
        "Respond with ONE line of strict JSON and nothing else:\n"
        '{"candidate_id": "<id>", "grade_llm": <0-4 integer>, "reason": "<one sentence '
        'grounded in the candidate facts>"}'
    )


def load_llm_grades(path):
    """Read eval/llm_grades.jsonl -> {candidate_id: (grade_int, reason)}.

    Tolerant: missing file -> {} (rule-only fallback); malformed/out-of-range
    lines are skipped. Never raises so the pipeline always completes."""
    path = Path(path)
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            cid = obj["candidate_id"]
            g = int(round(float(obj["grade_llm"])))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if 0 <= g <= 4:
            out[cid] = (g, str(obj.get("reason", "")).strip())
    return out


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def stream_candidates(path):
    """Yield candidate dicts line-by-line (memory-safe over the 465MB file)."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jd_profile(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_silver_set(candidates_path, jd, today, seed, total_target, llm_grades):
    """Two streaming passes -> the labelled silver records. Pure (no IO writes)."""
    # Pass 1: classify + bucket ids only.
    buckets, non_special = scan_and_bucket(stream_candidates(candidates_path), today)
    chosen = select_sample(buckets, non_special, seed=seed, total_target=total_target)
    id_to_stratum = {cid: s for s, ids in chosen.items() for cid in ids}
    selected = set(id_to_stratum)

    # Pass 2: pull the full records for exactly the selected ids.
    records = {}
    for c in stream_candidates(candidates_path):
        cid = c.get("candidate_id")
        if cid in selected:
            records[cid] = c
            if len(records) == len(selected):
                break

    labels = []
    for cid in sorted(selected):
        c = records[cid]
        grade_r, breakdown = grade_rules(c, jd, today)
        forced_zero = bool(breakdown["forced_zero"])
        llm = llm_grades.get(cid)
        grade_l = llm[0] if llm else None
        reason_l = llm[1] if llm else None
        final, needs_review = reconcile(grade_r, grade_l, forced_zero)
        labels.append({
            "candidate_id": cid,
            "stratum": id_to_stratum[cid],
            "grade_rules": grade_r,
            "grade_llm": grade_l,
            "grade_final": final,
            "needs_review": needs_review,
            "rule_breakdown": breakdown,
            "llm_reason": reason_l,
        })
    return labels, chosen


def write_outputs(labels, chosen, candidates_path, jd, today, seed, total_target):
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    OUT_LABELS.write_text(json.dumps(labels, indent=2), encoding="utf-8")

    OUT_SAMPLE_IDS.write_text(
        json.dumps(
            {
                "seed": seed,
                "total_target": total_target,
                "stratum_targets": STRATUM_TARGETS,
                "counts": {s: len(ids) for s, ids in chosen.items()},
                "ids_by_stratum": chosen,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # One LLM grading prompt per sampled candidate (ordered by id for stability).
    sample_ids = sorted(c for ids in chosen.values() for c in ids)
    records = {}
    want = set(sample_ids)
    for c in stream_candidates(candidates_path):
        if c.get("candidate_id") in want:
            records[c["candidate_id"]] = c
            if len(records) == len(want):
                break
    with OUT_LLM_PROMPTS.open("w", encoding="utf-8") as f:
        for cid in sample_ids:
            f.write(json.dumps({"candidate_id": cid, "prompt": build_llm_prompt(records[cid])}) + "\n")

    OUT_ANCHORS.write_text(json.dumps(build_anchors(), indent=2), encoding="utf-8")


def print_summary(labels, chosen, anchor_results, has_llm):
    def hist(key):
        h = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, None: 0}
        for x in labels:
            h[x[key]] = h.get(x[key], 0) + 1
        return h

    print("\n" + "=" * 70)
    print("SILVER LABEL SUMMARY")
    print("=" * 70)

    print(f"\nSampled {len(labels)} candidates (seed fixed, target {TOTAL_TARGET}).")
    print("\nStratum counts:")
    for s in ALL_STRATA:
        print(f"  {s:20s} {len(chosen.get(s, [])):4d}")

    print("\nGrade distribution (rule grader):")
    hr = hist("grade_rules")
    for g in (4, 3, 2, 1, 0):
        print(f"  grade {g}: {hr.get(g, 0):4d}")

    print(f"\nLLM grades present: {has_llm}")
    if has_llm:
        hl = hist("grade_llm")
        print("Grade distribution (LLM grader, where present):")
        for g in (4, 3, 2, 1, 0):
            print(f"  grade {g}: {hl.get(g, 0):4d}")
        print(f"  (no LLM grade): {hl.get(None, 0):4d}")

    print("\nFinal grade distribution:")
    hf = hist("grade_final")
    for g in (4, 3, 2, 1, 0):
        print(f"  grade {g}: {hf.get(g, 0):4d}")
    print(f"  null (needs review): {hf.get(None, 0):4d}")

    pairs = [(x["grade_rules"], x["grade_llm"]) for x in labels if x["grade_llm"] is not None]
    rep = agreement_report(pairs)
    print("\nRules-vs-LLM agreement:")
    if rep["n"] == 0:
        print("  (no LLM grades yet — run eval/llm_grading_prompts.jsonl offline,")
        print("   write eval/llm_grades.jsonl, and re-run for the agreement report)")
    else:
        print(f"  n (both present): {rep['n']}")
        print(f"  exact match:      {rep['exact_match']}")
        print(f"  within +/-1:      {rep['within_1']}")
        print(f"  spearman:         {rep['spearman']}")
        print(f"  kendall tau:      {rep['kendall']}")

    n_review = sum(1 for x in labels if x["needs_review"])
    print(f"\nNeeding human review (|rules - llm| >= 2): {n_review}")

    print("\nAnchor check (sacred — never tuned against):")
    n_ok = sum(1 for a in anchor_results if a["ok"])
    for a in anchor_results:
        flag = "ok " if a["ok"] else "XX "
        print(f"  [{flag}] {a['name']:34s} expected {a['expected']:>4s}  got {a['got']}")
    print(f"  anchors passed: {n_ok}/{len(anchor_results)}")
    print("=" * 70 + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the silver-standard relevance set (offline).")
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--jd-profile", default=str(DEFAULT_JD_PROFILE))
    parser.add_argument("--llm-grades", default=str(IN_LLM_GRADES))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--total", type=int, default=TOTAL_TARGET)
    args = parser.parse_args(argv)

    today = dt.date.fromisoformat(REFERENCE_DATE)
    jd = load_jd_profile(args.jd_profile)
    llm_grades = load_llm_grades(args.llm_grades)

    labels, chosen = build_silver_set(
        args.candidates, jd, today, args.seed, args.total, llm_grades
    )
    write_outputs(labels, chosen, args.candidates, jd, today, args.seed, args.total)
    anchor_results = check_anchors(jd, today)
    print_summary(labels, chosen, anchor_results, has_llm=bool(llm_grades))

    print(f"Wrote: {OUT_LABELS}")
    print(f"Wrote: {OUT_SAMPLE_IDS}")
    print(f"Wrote: {OUT_LLM_PROMPTS}  (run these offline -> {IN_LLM_GRADES})")
    print(f"Wrote: {OUT_ANCHORS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
