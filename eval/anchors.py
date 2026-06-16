"""STEP 5 — the sacred anchor set.

Tiny, hand-built, UNAMBIGUOUS cases defined in code: a clear honeypot MUST be 0,
a textbook Senior-AI-Engineer fit MUST be 4. These are NEVER tuned against — they
are a tripwire that the rule grader still agrees with obvious truth. Extremes use
an exact expected grade; near-boundary cases use a min/max range so the anchor
stays robust, not brittle.
"""

from __future__ import annotations

from eval.rubric import grade_rules


def _role(company, title, months, desc, current=False, start="2018-01-01", end="2020-01-01", size="501-1000"):
    return {
        "company": company,
        "title": title,
        "start_date": start,
        "end_date": None if current else end,
        "duration_months": months,
        "is_current": current,
        "industry": "Software",
        "company_size": size,
        "description": desc,
    }


def _cand(cid, title, yoe, roles, skills, location="Bangalore", country="India",
          company="Acme", headline="", summary="", github=40, relocate=True):
    return {
        "candidate_id": cid,
        "profile": {
            "anonymized_name": "Test Person",
            "headline": headline or title,
            "summary": summary,
            "location": location,
            "country": country,
            "years_of_experience": yoe,
            "current_title": title,
            "current_company": company,
            "current_company_size": "501-1000",
            "current_industry": "Software",
        },
        "career_history": roles,
        "education": [],
        "skills": skills,
        "redrob_signals": {
            "github_activity_score": github,
            "willing_to_relocate": relocate,
            "last_active_date": "2026-06-01",
            "recruiter_response_rate": 0.6,
            "open_to_work_flag": True,
            "interview_completion_rate": 0.9,
            "notice_period_days": 30,
        },
    }


def _sk(name, prof="advanced", months=24):
    return {"name": name, "proficiency": prof, "endorsements": 10, "duration_months": months}


def build_anchors():
    """~12 unambiguous cases. Each: {name, expected_grade?|expected_min/max, candidate}."""
    anchors = []

    # 1. Textbook Senior AI Engineer fit -> MUST be 4.
    anchors.append({
        "name": "textbook_senior_ai_engineer",
        "expected_grade": 4,
        "note": "right title + 7yr + product co + retrieval/ranking in prod + NLP/IR",
        "candidate": _cand(
            "ANCHOR_001", "Senior AI Engineer", 7.0,
            [
                _role("Flipkart", "Senior AI Engineer", 30,
                      "Built and deployed embeddings-based semantic search and a learning-to-rank "
                      "system for product search serving millions of users; measured NDCG and MRR "
                      "and ran A/B tests to improve relevance.", current=True),
                _role("Myntra", "Machine Learning Engineer", 36,
                      "Built a recommendation system and information-retrieval pipeline over text; "
                      "trained and deployed ranking models in production."),
            ],
            [_sk("NLP"), _sk("Information Retrieval"), _sk("Embeddings"), _sk("Learning to Rank")],
            summary="NLP/IR engineer building retrieval and ranking systems in production.",
        ),
    })

    # 2. Clear honeypot (impossible tenure) -> MUST be 0.
    anchors.append({
        "name": "honeypot_impossible_tenure",
        "expected_grade": 0,
        "note": "a single 120-month role but only 3 years total experience",
        "candidate": _cand(
            "ANCHOR_002", "Machine Learning Engineer", 3.0,
            [_role("Google", "ML Engineer", 120,
                   "Built ranking and retrieval systems in production.", current=True)],
            [_sk("NLP"), _sk("Ranking")],
        ),
    })

    # 3. Clear honeypot (expert in many skills, 0 months used) -> MUST be 0.
    anchors.append({
        "name": "honeypot_expert_zero_months",
        "expected_grade": 0,
        "note": "four expert skills with 0 months of usage",
        "candidate": _cand(
            "ANCHOR_003", "AI Engineer", 6.0,
            [_role("Amazon", "AI Engineer", 24,
                   "Built retrieval and ranking systems.", current=True)],
            [_sk("NLP", "expert", 0), _sk("Information Retrieval", "expert", 0),
             _sk("Ranking", "advanced", 0), _sk("Embeddings", "expert", 0)],
        ),
    })

    # 4. Keyword-stuffer (HR Manager stuffed with AI skills) -> MUST be 0.
    anchors.append({
        "name": "keyword_stuffer_hr_manager",
        "expected_grade": 0,
        "note": "non-tech HR title, many AI skills, zero career substance",
        "candidate": _cand(
            "ANCHOR_004", "HR Manager", 8.0,
            [_role("Infosys", "HR Manager", 48,
                   "Managed recruitment, onboarding, payroll and employee relations.", current=True)],
            [_sk("Machine Learning"), _sk("Deep Learning"), _sk("NLP"),
             _sk("Computer Vision"), _sk("LLMs")],
        ),
    })

    # 5. Pure noise (Accountant, no AI at all) -> MUST be 0.
    anchors.append({
        "name": "pure_noise_accountant",
        "expected_grade": 0,
        "note": "irrelevant profession, no AI/ML signal",
        "candidate": _cand(
            "ANCHOR_005", "Accountant", 9.0,
            [_role("Local Firm", "Accountant", 60,
                   "Prepared financial statements, tax filings and audits.", current=True)],
            [_sk("Excel"), _sk("Taxation"), _sk("Accounting")],
        ),
    })

    # 6. Career entirely at consulting/services -> force low (cap 1).
    anchors.append({
        "name": "consulting_only_services",
        "expected_max": 1,
        "note": "real ML title but career entirely at TCS/Infosys/Wipro",
        "candidate": _cand(
            "ANCHOR_006", "Machine Learning Engineer", 7.0,
            [
                _role("TCS", "Machine Learning Engineer", 36,
                      "Built ML models and retrieval pipelines for client projects.", current=True),
                _role("Infosys", "Data Scientist", 36,
                      "Worked on text ranking and recommendation for clients."),
            ],
            [_sk("NLP"), _sk("Information Retrieval")],
            company="TCS",
        ),
    })

    # 7. Primary CV/speech, no NLP/IR -> force low (cap 1).
    anchors.append({
        "name": "cv_speech_no_nlp",
        "expected_max": 1,
        "note": "strong title but computer-vision/speech only, no NLP/IR",
        "candidate": _cand(
            "ANCHOR_007", "AI Engineer", 7.0,
            [_role("Bosch", "AI Engineer", 48,
                   "Built computer vision object-detection and image classification models, plus "
                   "speech recognition (ASR) for in-car systems. Focused on vision and audio, not text.",
                   current=True)],
            [_sk("Computer Vision"), _sk("Image Classification"), _sk("Speech Recognition")],
        ),
    })

    # 8. Strong adjacent "hidden gem" (Data Engineer doing retrieval/ranking) -> 3-4.
    anchors.append({
        "name": "adjacent_gem_data_engineer",
        "expected_min": 3,
        "expected_max": 4,
        "note": "plain title, but descriptions show real retrieval+ranking in production",
        "candidate": _cand(
            "ANCHOR_008", "Data Engineer", 7.0,
            [
                _role("Swiggy", "Data Engineer", 30,
                      "Built the semantic search and embeddings retrieval backend for restaurant "
                      "search; implemented a learning-to-rank model and measured NDCG, deployed to "
                      "production serving millions of users.", current=True),
                _role("Ola", "Software Engineer", 30,
                      "Built recommendation and information-retrieval systems over text in production."),
            ],
            [_sk("Embeddings"), _sk("Elasticsearch"), _sk("Ranking")],
            summary="Data/backend engineer who built retrieval and ranking systems.",
        ),
    })

    # 9. Too junior (2 yrs) ML engineer -> weak/partial (1-2).
    anchors.append({
        "name": "too_junior_ml",
        "expected_min": 1,
        "expected_max": 2,
        "note": "real ML work but only 2 years of experience",
        "candidate": _cand(
            "ANCHOR_009", "Machine Learning Engineer", 2.0,
            [_role("Startup", "ML Engineer", 24,
                   "Built and deployed a retrieval and ranking model in production.", current=True)],
            [_sk("NLP", "intermediate", 18), _sk("Information Retrieval", "intermediate", 18)],
        ),
    })

    # 10. Good fit with a minor gap (slightly senior, otherwise strong) -> 3-4.
    anchors.append({
        "name": "good_fit_minor_gap",
        "expected_min": 3,
        "expected_max": 4,
        "note": "strong substance, 9 yrs (top of band)",
        "candidate": _cand(
            "ANCHOR_010", "Machine Learning Engineer", 9.0,
            [
                _role("Zomato", "Machine Learning Engineer", 40,
                      "Built embeddings retrieval and learning-to-rank for search; deployed to "
                      "production, measured NDCG/MRR, ran A/B tests.", current=True),
                _role("Paytm", "Data Scientist", 40,
                      "Built recommendation and NLP ranking systems in production."),
            ],
            [_sk("NLP"), _sk("Embeddings"), _sk("Ranking")],
        ),
    })

    # 11. Adjacent title, NO substance -> partial/weak (1-2).
    anchors.append({
        "name": "adjacent_no_substance",
        "expected_min": 1,
        "expected_max": 2,
        "note": "backend engineer with only CRUD/web work, no retrieval/ranking",
        "candidate": _cand(
            "ANCHOR_011", "Backend Engineer", 7.0,
            [_role("SomeCo", "Backend Engineer", 48,
                   "Built REST APIs, CRUD services and payment integrations with Spring Boot.",
                   current=True)],
            [_sk("Java"), _sk("Spring Boot"), _sk("PostgreSQL")],
        ),
    })

    # 12. Non-India, not relocating, otherwise strong -> down-weighted but real (2-3).
    anchors.append({
        "name": "strong_but_non_india_no_relocate",
        "expected_min": 2,
        "expected_max": 3,
        "note": "strong substance but USA-based and not willing to relocate",
        "candidate": _cand(
            "ANCHOR_012", "Senior AI Engineer", 7.0,
            [_role("Stripe", "Senior AI Engineer", 40,
                   "Built embeddings retrieval and learning-to-rank search in production; NDCG/MRR.",
                   current=True)],
            [_sk("NLP"), _sk("Embeddings"), _sk("Ranking")],
            location="San Francisco", country="USA", relocate=False,
        ),
    })

    return anchors


def check_anchors(jd, today):
    """Grade every anchor and report pass/fail against its expected grade/range."""
    results = []
    for a in build_anchors():
        g, _ = grade_rules(a["candidate"], jd, today)
        if "expected_grade" in a:
            ok = g == a["expected_grade"]
            expect = str(a["expected_grade"])
        else:
            lo = a.get("expected_min", 0)
            hi = a.get("expected_max", 4)
            ok = lo <= g <= hi
            expect = f"{lo}-{hi}"
        results.append({"name": a["name"], "expected": expect, "got": g, "ok": ok, "note": a["note"]})
    return results
