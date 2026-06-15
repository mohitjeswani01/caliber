# Caliber — Architecture & Interface Contracts
 
This is the engineering blueprint. `STRATEGY.md` says *what wins and why*; this
says *how the code fits together*. Every module's input/output contract is fixed
here so we can build in parallel without collisions. If a contract needs to
change, change it here first, then the code.
 
---
 
## 1. The two-phase pipeline
 
```
                    OFFLINE  (scripts/precompute.py — no time limit)
  ┌────────────────────────────────────────────────────────────────────┐
  │  job_description.docx ──► jd_profile.py ──► artifacts/jd_profile.json │
  │  candidates.jsonl ──► embeddings.py ──► artifacts/candidate_emb.npy   │
  │                              │                                         │
  │                              └──► index.py ──► artifacts/faiss.index   │
  │  (cross-encoder + LTR model weights downloaded ──► models/)           │
  └────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ONLINE  (rank.py — ≤5 min, CPU, no network)
  ┌────────────────────────────────────────────────────────────────────┐
  │  load candidates (stream) + artifacts                                 │
  │     │                                                                  │
  │     ├─ semantic score  (query emb vs faiss.index)      [retrieval]    │
  │     ├─ lexical score   (BM25 over role descriptions)   [retrieval]    │
  │     │        └─► fuse to a candidate shortlist (~800)  fusion.py      │
  │     ├─ structured features (features.py, skill-gated)                 │
  │     ├─ honeypot filter  (honeypot.py ──► floor)                       │
  │     ├─ cross-encoder rerank on shortlist (cross_encoder.py)           │
  │     ├─ LTR model combines all signals (ltr.py)                        │
  │     └─ behavioral multiplier (behavioral.py)                          │
  │          ▼                                                             │
  │     scorer.py ──► composite ──► ranker.py ──► top 100                  │
  │          ▼                                                             │
  │     reasoning.py (grounded templates) ──► submission.csv + validate   │
  └────────────────────────────────────────────────────────────────────┘
```
 
**Hard rule restated:** everything in OFFLINE may use network/models freely.
Everything in ONLINE is CPU-only, no network, no hosted LLM, ≤5 min, ≤16 GB.
 
---
 
## 2. Canonical data types (src/caliber/schema.py)
 
The single source of truth for candidate structure. Built from
`data/challenge/candidate_schema.json`. Everything else imports these.
 
```python
@dataclass
class Candidate:
    candidate_id: str
    profile: Profile               # headline, summary, yoe, current_title, ...
    career_history: list[Role]     # company, title, dates, duration_months, description, ...
    education: list[Education]
    skills: list[Skill]            # name, proficiency, endorsements, duration_months
    certifications: list[Cert]
    languages: list[Language]
    redrob_signals: Signals        # the 23 behavioral signals
    raw: dict                      # keep the original dict for reasoning/debug
```
 
Contract: `schema.parse_candidate(dict) -> Candidate` and
`schema.candidate_to_text(Candidate) -> str` (the rich text used for embeddings —
headline + summary + role titles + role **descriptions** + skills-with-context).
 
---
 
## 3. Module contracts (the interfaces that must not drift)
 
### OFFLINE modules
 
**`jd_profile.py`** — *(Mohit owns)*
- `build_jd_profile(jd_text: str) -> dict` → writes `artifacts/jd_profile.json`.
- Output schema (aspect-based, this is the key design):
```json
  {
    "role": "Senior AI Engineer",
    "experience_band": {"min": 5, "max": 9, "ideal_min": 6, "ideal_max": 8},
    "aspects": {
      "embeddings_retrieval": {"weight": 0.0, "query_text": "...", "keywords": [...]},
      "vector_db_search":     {"weight": 0.0, "query_text": "...", "keywords": [...]},
      "ranking_eval":         {"weight": 0.0, "query_text": "...", "keywords": [...]},
      "nlp_ir_background":    {"weight": 0.0, "query_text": "...", "keywords": [...]},
      "product_company":      {"weight": 0.0, "query_text": "...", "keywords": [...]},
      "production_recency":   {"weight": 0.0, "query_text": "...", "keywords": [...]}
    },
    "disqualifiers": ["pure_research_no_prod", "consulting_only", "cv_speech_robotics_no_nlp", "title_chaser", "langchain_only_recent"],
    "location_prefs": {"preferred_cities": [...], "country_priority": "India", "relocation_ok": true},
    "consulting_firms": ["TCS","Infosys","Wipro","Accenture","Cognizant","Capgemini","Mindtree","LTIMindtree","HCL","Tech Mahindra"]
  }
```
- Each aspect has a `query_text` that gets embedded separately (aspect-based
  matching). Weights are placeholders here; the LTR model learns real ones.
**`embeddings.py`** — *(friend owns)*
- `load_model() -> SentenceTransformer` — loads a SMALL local model
  (`BAAI/bge-small-en-v1.5`, 384-dim). Model cached under `models/`.
- `encode_candidates(candidates: Iterable[Candidate], batch_size=256) -> np.ndarray`
  → returns float32 array shape `(N, 384)`, L2-normalized. Saved by precompute to
  `artifacts/candidate_emb.npy`. **Row i corresponds to the i-th candidate in
  file order** — order is the join key, so it must be stable.
- `encode_texts(texts: list[str]) -> np.ndarray` — same encoder, for JD aspect
  queries.
- Also writes `artifacts/candidate_ids.npy` (the ordered id list) so any consumer
  can map row index → candidate_id without re-reading the 465MB file.
**`index.py`** — *(friend owns)*
- `build_index(emb: np.ndarray) -> faiss.Index` (IndexFlatIP on normalized vectors
  = cosine). `save_index(index, path)` / `load_index(path)`.
- `search(index, query_emb: np.ndarray, k: int) -> (scores, indices)`.
### ONLINE modules
 
**`io_utils.py`** — *(Mohit owns)* — `stream_candidates(path) -> Iterator[Candidate]`
(memory-safe, line by line); `load_all_ids(path) -> list[str]`.
 
**`features.py`** — *(Mohit owns)* — `structured_features(c: Candidate, jd: dict) -> dict[str,float]`.
Includes the **skill-gating** logic: a skill's credit is multiplied by how well
the career history corroborates it. Returns named features for the LTR model.
 
**`honeypot.py`** — *(Mohit owns, already drafted Phase 1)* —
`is_honeypot(c: Candidate) -> tuple[bool, list[str]]`. Detected → score floor.
 
**`behavioral.py`** — *(friend owns)* — `behavioral_multiplier(c: Candidate) -> float`
in a bounded range (e.g. 0.5–1.15) from the 23 signals.
 
**`cross_encoder.py`** — *(friend owns)* — `rerank(jd_text, candidates_subset) -> scores`.
Local cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`), runs only on the
~800 shortlist, CPU. Weights under `models/`.
 
**`fusion.py`** — *(friend owns)* — `reciprocal_rank_fusion(*rankings, k=60) -> fused`.
 
**`ltr.py`** — *(friend owns, later)* — `train(...)` offline (scripts/train_ltr.py),
`predict(features: np.ndarray) -> np.ndarray` online. LightGBM ranker. Model →
`models/ltr.txt`.
 
**`scorer.py`** — *(Mohit owns)* — combines everything into one composite per
candidate. The orchestration policy lives here.
 
**`reasoning.py`** — *(Mohit owns)* — `reason(c, jd, feature_values) -> str`.
Grounded templates over REAL fields + aspect scores. No invented facts. Honest
about gaps. Tone matches rank.
 
**`ranker.py`** — *(Mohit owns)* — the orchestrator `rank.py` calls. Produces the
final top-100, enforces non-increasing score + candidate_id tie-break, writes CSV,
self-validates with `tests/validate_submission.py`.
 
### EVAL modules *(built together, right after)*
 
**`eval/metrics.py`** — `ndcg_at_k`, `map_score`, `precision_at_k`, `composite`.
**`eval/evaluate.py`** — runs a ranking against silver labels, prints the composite
+ per-component breakdown + an ablation table.
**`scripts/make_silver_labels.py`** — offline, builds the silver relevance set.
---
 
## 4. Artifact formats (the offline → online handoff)
 
| File | Written by | Read by | Format |
|---|---|---|---|
| `artifacts/jd_profile.json` | jd_profile.py | features, scorer, reasoning | JSON (Section 3) |
| `artifacts/candidate_emb.npy` | embeddings.py | index, ranker | float32 (N, 384), normalized |
| `artifacts/candidate_ids.npy` | embeddings.py | ranker | str array, length N, file order |
| `artifacts/faiss.index` | index.py | ranker | FAISS IndexFlatIP |
| `models/` (bge, cross-encoder, ltr) | precompute / train_ltr | embeddings, cross_encoder, ltr | model files |
 
All gitignored. `scripts/precompute.py` produces all `artifacts/`; document its
runtime in `submission_metadata.yaml` (pre-computation is allowed to exceed 5 min;
only `rank.py` is bound by the budget).
 
---
 
## 5. Determinism & reproducibility rules
 
- Fixed seeds everywhere (`config.SEED = 42`).
- Stable sorts; final tie-break by `candidate_id` ascending.
- `score` column strictly non-increasing with rank.
- `rank.py` must run end-to-end from artifacts in ≤5 min on 16 GB CPU, no network.
- Every CSV self-validated before it's considered done.
---
 
## 6. Ownership split (for parallel work + Stage-5 defense)
 
- **Mohit (pipeline / scoring / logic):** schema, io_utils, jd_profile, features,
  honeypot, scorer, reasoning, ranker.
- **Friend (ML / retrieval / eval-infra):** embeddings, index, cross_encoder,
  fusion, ltr, behavioral.
- Eval harness (metrics, evaluate, silver labels): built together — it's the
  shared measuring stick.
- Each person reviews the other's PRs and must be able to explain their half cold.