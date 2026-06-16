# Caliber

Caliber is our entry to the Redrob **"India Runs" Track 1 — Intelligent Candidate
Discovery & Ranking Challenge**. It ranks the **top 100** best-fit candidates out
of a **100,000**-profile pool for a *Senior AI Engineer* job description — by
reading career **substance**, not matching keywords.

The dataset is adversarial: keyword stuffers, plain-language strong fits, behavioral
twins, and ~80 internally-impossible honeypots are planted to defeat the naive
"embed the JD, sort by cosine" baseline. Caliber is built to see through all four.

- **What & why:** [`CLAUDE.md`](CLAUDE.md) (the operating rules) and
  [`docs/STRATEGY.md`](docs/STRATEGY.md) (the competitive intelligence).
- **Scoring:** `0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10` — the top ~50
  picks dominate the score.

## Architecture — two strictly-separated phases

1. **OFFLINE** (`scripts/precompute.py`, no time limit) — parse the JD into a
   requirement profile, encode all 100K candidates with a small local
   sentence-transformer, build a FAISS index, and persist everything to
   `artifacts/`. May use an LLM for offline silver-label generation only.
2. **ONLINE** (`rank.py`, the budget below) — load artifacts → hybrid score
   (`semantic + BM25 + gated structured features`) `× behavioral multiplier`,
   honeypots forced to the floor → select top 100 → grounded template reasoning
   → write + self-validate the CSV. Pure CPU, deterministic, no network.

### Hard constraints on the ranking step (non-negotiable — Stage-3 disqualifier)

| Constraint | Limit |
|---|---|
| Runtime | ≤ 5 minutes wall-clock |
| Memory | ≤ 16 GB RAM |
| Compute | CPU only — no GPU |
| Network | Off — no hosted LLM / no model downloads at runtime |
| Disk | ≤ 5 GB intermediate state |

No per-candidate LLM call ever lives in the ranking path.

## Reproduce

```bash
# 1. Environment (Python 3.11, CPU-only)
python -m venv .venv && source .venv/bin/activate

# 1a. Install the CPU-only torch wheel explicitly from the PyTorch CPU index.
#     This guarantees the identical 2.2.2+cpu wheel (no CUDA deps) at Stage-3
#     reproduction. Do this BEFORE the requirements file.
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

# 1b. Install the rest. requirements.txt also carries an --extra-index-url for
#     torch, so this line alone is self-sufficient; running 1a first just makes
#     the CPU wheel unambiguous.
pip install -r requirements.txt

# 2. OFFLINE precompute — builds artifacts/ (embeddings, FAISS index, JD profile).
#    No time limit; run once. Records precompute time to submission_metadata.yaml.
python scripts/precompute.py --candidates ./data/candidates.jsonl

# 3. ONLINE ranking — the single command Stage 3 reproduces (≤5 min, CPU, no network):
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv

# 4. Validate the output before it counts as done:
python tests/validate_submission.py ./submission.csv
```

> The ranking step (step 3) is what must fit the 5-minute / 16 GB / CPU-only
> budget. Precompute (step 2) may take longer and is run ahead of time; its
> outputs in `artifacts/` are static inputs to `rank.py`.

## Repository layout

```
src/caliber/        # the package (offline builders + online scoring modules)
scripts/            # precompute, data profiling, silver labels, LTR training (offline)
eval/               # NDCG/MAP/P@k metrics + offline evaluation harness
sandbox/            # hosted small-sample demo app (Stage-1 sandbox requirement)
tests/              # schema / honeypot / metrics / CSV-format tests + validator
docs/               # STRATEGY.md (and DEFENSE.md — design rationale for Stage 5)
rank.py             # thin online entry point → src/caliber/ranker.py
data/, artifacts/   # gitignored (large; not committed)
```

## Status

Scaffold only — module boundaries and docstrings are in place; ranking logic is
built incrementally from here, with tests and commits per logical change.
