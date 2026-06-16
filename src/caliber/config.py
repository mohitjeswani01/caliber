"""Central configuration: paths, model names, weights, and tunable constants.
Single source of truth for everything the rest of the package needs to agree on:
- Filesystem layout (``data/``, ``artifacts/``, ``models/`` locations).
- The local sentence-transformer model id used offline AND online (must be the
  same model, cached locally, so ``rank.py`` makes no network call).
- Hybrid-score component weights and the behavioral-multiplier envelope bounds
  (Section 4 & 6 of STRATEGY.md). These are *defaults* to be tuned against the
  silver labels, not frozen guesses.
- Determinism knobs: the global random seed and the canonical tie-break key.
Keeping these out of the logic modules makes the weights easy to sweep during
tuning and keeps ``rank.py`` reproducible.
"""
from __future__ import annotations

from pathlib import Path

# --- determinism knobs (ARCHITECTURE.md §5) ---------------------------------
# The single global random seed. Every stochastic step in the project (silver
# sampling, any train/test split, LTR training) reads this so two runs are
# bit-identical. The canonical final tie-break is ``candidate_id`` ascending.
SEED = 42
TIE_BREAK_KEY = "candidate_id"

# The dataset's reference "now". The pool is a static snapshot, so honeypot
# date-consistency checks and recency logic must compare against a FIXED date
# (never the wall clock) to stay deterministic and reproducible. This matches
# the snapshot the candidates.jsonl pool was generated against.
REFERENCE_DATE = "2026-06-16"

# --- filesystem layout -----------------------------------------------------
# config.py lives at src/caliber/config.py, so parents[2] is the repo root.
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ROOT / "models"
CANDIDATES_PATH = DATA_DIR / "candidates.jsonl"

# --- embedding model (the SAME model offline + online) ---------------------
# Small, 384-dim, CPU-friendly. Downloaded once during offline precompute and
# saved under MODELS_DIR so the online ranker never touches the network.
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_MODEL_LOCAL_DIR = MODELS_DIR / "bge-small-en-v1.5"
EMBED_DIM = 384
# Truncate candidate text to this many tokens before encoding. Quadratic
# attention on CPU makes throughput highly sensitive to sequence length; on the
# dev box 256 encodes ~2.5x faster than the model's 512 default while still
# covering headline + summary + the current-role description (the signal that
# surfaces plain-language Tier-5s). Candidate text averages ~440 tokens, so this
# does truncate older roles; raise toward 512 for max fidelity if compute allows.
EMBED_MAX_SEQ_LENGTH = 256
# bge retrieval is asymmetric: the instruction is prepended to the QUERY side
# (the JD aspect queries) only, never to the candidate "passages". See bge model
# card. Applied in embeddings.encode_texts(is_query=True).
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# --- artifact filenames (offline -> online handoff; ARCHITECTURE.md §4) -----
CANDIDATE_EMB_FILE = "candidate_emb.npy"     # float32 (N, EMBED_DIM), L2-normalized
CANDIDATE_IDS_FILE = "candidate_ids.npy"     # str array, length N, candidate file order
FAISS_INDEX_FILE = "faiss.index"             # FAISS IndexFlatIP over the embeddings
JD_PROFILE_FILE = "jd_profile.json"          # JD requirement profile (jd_profile.py)