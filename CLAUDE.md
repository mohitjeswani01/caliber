# CLAUDE.md — Caliber
 
Project memory for **Caliber**, our entry to the Redrob "India Runs" Track 1:
the *Intelligent Candidate Discovery & Ranking Challenge*.
 
Read **@docs/STRATEGY.md** for the full competitive intelligence: the relevance
model, scoring breakdown, traps, and architecture rationale. This file is the
operating constitution (the *rules*); STRATEGY.md is the *why*.
 
## Mission
Rank the **top 100** best-fit candidates out of a **100,000** pool for a
*Senior AI Engineer* job description — the way a great recruiter would: by
reading career **substance**, not matching keywords. Produce a validated CSV.
The score is dominated by **NDCG@10 (50%)**, then NDCG@50 (30%), MAP (15%),
P@10 (5%). Getting the top ~50 surgically right is the whole game.
 
## The one rule that wins
**Career substance > keyword presence.** A profile that *lists* "RAG / LLM /
NLP" is worth nothing unless the career history shows real retrieval, ranking,
recommendation, or applied-ML work at a **product** company. The dataset is
adversarial and engineered to punish keyword matching. If a scoring decision
ever rewards a keyword without corroborating career evidence, it is wrong.
 
## Hard constraints — NON-NEGOTIABLE (violation = Stage 3 disqualification)
The ranking step (`rank.py`) MUST:
- Finish in **≤ 5 minutes** wall-clock
- Use **≤ 16 GB RAM**
- Run **CPU-only** (no GPU)
- Make **ZERO network calls** — no OpenAI / Anthropic / Cohere / Gemini / any
  hosted LLM, no model downloads at runtime
- Use **≤ 5 GB** disk for intermediate state
Therefore: **never make a per-candidate LLM call in the ranking path.** LLMs may
be used **offline only** (eval-label generation, design discussion), with their
outputs baked into static artifacts that `rank.py` reads. 100K hosted-LLM calls
cannot fit the budget — this constraint is the filter that eliminates lazy
submissions, and it is our advantage if we respect it.
 
## Architecture — two phases, kept strictly separate
1. **OFFLINE** (`scripts/precompute.py`, no time limit): build the JD
   requirement profile; encode all 100K candidates with a **small local
   sentence-transformer** (e.g. `BAAI/bge-small-en-v1.5` or `intfloat/e5-small-v2`);
   build a FAISS index; persist everything to `artifacts/`. Record precompute
   time in `submission_metadata.yaml`.
2. **ONLINE** (`rank.py`, the 5-min budget): load artifacts → hybrid score →
   select top 100 → generate grounded reasoning → write CSV. Pure CPU,
   fully deterministic.
Hybrid score (see @docs/STRATEGY.md for weights and the relevance model):
`semantic_similarity + lexical(BM25) + gated_structured_features` then
`× behavioral_multiplier`, with detected honeypots forced to the score floor.
 
## The traps (dataset is adversarial — design against each)
- **Keyword stuffers**: non-tech title + many AI skills. *Gate* skill credit
  behind title/career substance — a skill only counts if the history backs it.
- **Plain-language Tier-5s**: genuine fit, no buzzwords. Surface them via
  semantic understanding of role **descriptions**, not skill tags.
- **Behavioral twins**: near-identical profiles, different behaviour. The
  behavioral multiplier breaks them apart.
- **~80 honeypots**: impossible profiles (tenure exceeding company age; "expert"
  in many skills with 0 months used). Forced to relevance tier 0 in ground
  truth. **> 10% honeypot rate in our top 100 = disqualified.** Build a
  consistency detector; force detected honeypots to the score floor.
## Reasoning column (judged at Stage 4 manual review)
Generate every reasoning string **deterministically from extracted profile
facts** — never invent. Each entry must: cite specific facts (years, current
title, named skills, signal values); connect to a specific JD requirement;
honestly acknowledge real gaps; differ meaningfully from other entries; and
match its tone to the rank. **Any claim not present in the candidate's profile
is a hallucination and is penalized.** No LLM at runtime — pure templating over
real fields is safer here precisely *because* it cannot hallucinate.
 
## Engineering conventions
- Python 3.11. Deterministic everywhere: fixed seeds, stable sorts, ties broken
  by `candidate_id` ascending.
- Memory-safe: stream `candidates.jsonl` line-by-line; do not naively load all
  465 MB into memory where it can be avoided.
- Single reproduce command:
  `python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv`
- Validate **every** output CSV with the provided `validate_submission.py`
  before considering a task done.
- Tests required for: schema loading, honeypot detector, metrics
  (NDCG/MAP/P@k), and CSV format.
- **Commit incrementally** with clear, specific messages — one logical change
  per commit. Stage 4 inspects git history for genuine iteration; a single
  giant dump is a red flag. This matters as much as the code.
## How to work with me (the human)
- Inspect/research before building. Propose a short plan and wait for my go on
  anything structural or architectural.
- Work in small, reviewable steps. After each module, explain in plain language
  what you built and why — I have to defend every design choice in a live
  Stage 5 interview, so I must understand it, not just receive it.
- If you are unsure whether something fits the compute budget or the relevance
  model, stop and flag it rather than guessing.
## Do NOT
- Do NOT commit `data/` or `artifacts/` (huge — keep them gitignored).
- Do NOT rank by raw embedding cosine similarity alone — that is the losing
  baseline the dataset is built to defeat.
- Do NOT special-case the public sample to inflate a score; build a general
  system.
- Do NOT place secrets or API keys anywhere in the repo.