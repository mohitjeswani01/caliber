# Caliber — Strategy & Competitive Intelligence
 
This is the "why" behind every rule in `CLAUDE.md`. It is derived from a full
read of the challenge bundle (submission spec, JD, signals doc, schema) and a
profile of all 100,000 candidates in the released pool.
 
---
 
## 1. What this challenge actually is
 
It is **not** a skill-matching problem. It is a **fit-discernment** problem
disguised as one. The released `sample_submission.csv` is a deliberately *bad*
ranking: it puts an HR Manager, a Content Writer, and a Graphic Designer in the
top 6 — purely because they stuffed AI keywords into their skills list. That is
the default output of "embed the JD, embed the profile, sort by cosine," and it
is engineered to **lose**.
 
The win condition: a system that reads **career substance over keyword
presence**, surfaces genuine fits who never use the buzzwords, and rejects
impostors who use all of them.
 
---
 
## 2. The pool, by the numbers (full 100K profile)
 
- **Relevant candidates are rare by design.** ~0.73% have any ML/AI title at
  all. Counts: ML Engineer 167, AI Research Engineer 153, Data Scientist 145,
  Junior ML Engineer 131, "Senior Software Engineer (ML)" 142, AI Specialist
  130; the genuinely senior ones are tiny (Senior AI Engineer 4, Senior ML
  Engineer 6, Staff ML Engineer 6). The JD confirms this is intentional:
  "we'd rather see 10 great matches than 1000 maybes."
- **~68% is pure noise**: Business Analyst, HR Manager, Mechanical Engineer,
  Accountant, Sales, etc. (~5,700 each). These exist to be filtered; the
  keyword-stuffers among them are the bait.
- **Hidden gems live in adjacent titles**: Data Engineer 744, Senior Data
  Engineer 687, Analytics Engineer 764, Backend/Software Engineers whose
  *career descriptions* reveal real retrieval/ranking/recommender work. These
  are the "Tier-5 who didn't say the buzzword" the JD explicitly wants surfaced.
- **Geography**: ~75% India, ~10% USA, rest scattered (Australia, Canada, UK,
  Germany, Singapore, UAE ~2.5K each). JD wants India Tier-1 (Pune, Noida,
  Hyderabad, Mumbai, Delhi NCR, Bangalore) or relocation-willing; non-India is
  "case-by-case, no visa sponsorship" → down-weight heavily.
- **Experience**: median ~6.8 yrs; ~34% fall in the 5–9 band the JD targets.
- **Behavioral signals are real**: recruiter_response_rate median 0.44; ~35%
  open_to_work; ~65% have no GitHub (`github_activity_score == -1`); most active
  within the last few months. An inactive, unresponsive, perfect-on-paper
  candidate is "not actually available" → down-weight.
---
 
## 3. Scoring, decoded
 
`Final = 0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10`,
computed once against hidden ground truth after close. No leaderboard, no
feedback, **3 submissions max**.
 
Implications:
- **Top 10 = half the score.** Top 50 = another 30%. Ranks 51–100 barely move
  the needle except via MAP. Spend effort proportionally: the top ~50 must be
  near-perfect; one honeypot or stuffer in the top 10 is catastrophic.
- We must rank **exactly 100**, each rank 1–100 once, `score` non-increasing,
  ties broken by `candidate_id` ascending, filename = our participant/team id,
  CSV/UTF-8. Run `validate_submission.py` before every upload.
---
 
## 4. The relevance model (what ground truth almost certainly rewards)
 
Reverse-engineered from the JD. Use this as the structured-feature scoring
backbone. Suggested starting weights — to be **tuned against our own silver
labels (Section 7), not guessed-and-frozen**:
 
1. **Role substance (dominant)** — career history shows building retrieval /
   ranking / recommendation / search / applied-ML systems. Semantic + structured,
   never keyword. ~35%.
2. **Experience band** — 5–9 yrs, ideal 6–8; penalize both too-junior and
   too-senior. ~10%.
3. **NLP/IR over CV/speech/robotics** — primary CV/speech/robotics without
   NLP/IR is an explicit negative. ~10%.
4. **Product-company experience**, not career-long services/consulting
   (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, Mindtree/LTIMindtree,
   HCL, Tech Mahindra). Career *entirely* at these = strong negative. ~10%.
5. **Recent production code** — penalize 18+ months in pure "architecture /
   tech-lead" with no recent shipping. ~5%.
6. **Not a title-chaser** — penalize avg tenure < ~1.5 yrs across roles. ~5%.
7. **External validation** — open source / GitHub activity is a positive. ~5%.
8. **Location** — India Tier-1 or relocation-willing; non-India down-weight. ~10%.
9. **Behavioral multiplier** (Section 6) — applied multiplicatively on top.
These are the *base relevance* drivers; the behavioral multiplier modulates the
final score. Exact split is an empirical question — tune it.
 
### Explicit JD disqualifiers (force low)
- Pure research / academic-only, no production deployment.
- "AI experience" = only <12-month LangChain-calls-OpenAI, with no pre-LLM ML.
- Senior who hasn't shipped code in 18+ months.
- Title-chaser (company switch every ~1.5 yrs).
- Career entirely at consulting/services firms.
- Primary CV/speech/robotics without NLP/IR.
- Entirely closed-source proprietary 5+ yrs with no external validation.
---
 
## 5. The four traps and how Caliber beats each
 
| Trap | Signature | Defense |
|---|---|---|
| Keyword stuffer | Non-tech title + many AI skills | Gate skill credit behind title/career substance; a skill counts only if the role history corroborates it |
| Plain-language Tier-5 | Real fit, zero buzzwords | Embed and reason over role **descriptions**, not skill tags |
| Behavioral twin | Identical profile, different behaviour | Behavioral multiplier separates them |
| Honeypot (~80) | Impossible profile internals | Consistency detector → force to score floor |
 
---
 
## 6. Behavioral multiplier (the 23 redrob_signals)
 
Apply as a bounded multiplier (e.g. 0.5–1.15) on the base relevance score, so a
strong-on-paper but unavailable candidate is pushed down without being erased:
- Positive: recent `last_active_date`, `open_to_work_flag`, high
  `recruiter_response_rate`, `saved_by_recruiters_30d`, high
  `interview_completion_rate`, `profile_completeness_score`, verified
  email/phone, healthy `github_activity_score`.
- Negative: stale last-active (months), very low response rate, low interview
  completion, very long `notice_period_days`.
- Use envelopes/normalization, not raw values; keep it bounded so behaviour
  modulates rather than dominates the substance score.
---
 
## 7. Our edge: build our own ground truth
 
No labels, no leaderboard, 3 submissions. Most teams fly blind. We will not.
 
**Offline**, build a **silver-standard relevance set**: take a stratified sample
(strong ML titles, adjacent titles, noise, suspected stuffers, suspected
honeypots), score each against a JD-derived rubric (an LLM may assist here —
this is offline, not the ranking path), and assign silver relevance tiers.
Implement NDCG@10/@50, MAP, P@10 in `eval/metrics.py` and tune the ranker's
weights to maximize the composite against this silver set — **without
overfitting to the LLM's taste**. This converts "guess and hope" into "measure
and optimize," and is how we know we are winning *before* submitting.
 
Sanity checks to automate: no honeypots in our top 100; no non-tech
keyword-stuffer in the top 50; known plain-language fits do surface.
 
---
 
## 8. The pipeline (concrete)
 
**Offline (`scripts/precompute.py`):**
1. Parse JD → structured requirement profile (`jd_profile.json`): must-haves,
   nice-to-haves, disqualifiers, location prefs, experience band.
2. Build a rich text representation per candidate (headline + summary + role
   titles + role **descriptions** + skills-with-context).
3. Encode with a small local sentence-transformer → `candidate_embeddings.npy`.
4. Build FAISS index → `faiss.index`. Persist all to `artifacts/`.
**Online (`rank.py`, ≤5 min CPU):**
1. Load candidates (streaming) + artifacts.
2. Semantic score (query embedding vs index) + lexical BM25 over descriptions.
3. Structured feature score (Section 4), with skill-gating.
4. Honeypot detection → floor.
5. Behavioral multiplier (Section 6).
6. Composite → select top 100 → non-increasing scores, deterministic tie-break.
7. Grounded template reasoning per candidate from extracted facts.
8. Write + self-validate the CSV.
---
 
## 9. Stages 3–5: the human gauntlet (why we build for real)
 
Top submissions face: code reproduction in a constrained sandbox; **git-history
authenticity** (genuine iteration vs single dump); methodology review; reasoning
quality; and a **30-minute defend-your-work video interview** with Redrob
engineering. The eval is explicitly designed so AI-assisted-but-real-engineering
wins and AI-only loses. Therefore: commit incrementally, understand every
module, and keep `docs/DEFENSE.md` updated with the rationale for each design
choice as we make it. The interview is where most teams fail — and where we win.