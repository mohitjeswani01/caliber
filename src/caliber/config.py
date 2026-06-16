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
# The single global random seed. Every stochastic step (silver sampling, any
# train/test split, LTR training) reads this so two runs are bit-identical.
SEED = 42
TIE_BREAK_KEY = "candidate_id"

# The dataset's reference "now". The pool is a static snapshot, so honeypot
# date-consistency checks and recency logic compare against a FIXED date (never
# the wall clock) to stay deterministic and reproducible.
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
# Truncate candidate text to this many tokens before encoding (separate from the
# cross-encoder cap below). 256 encodes ~2.5x faster than the 512 default while
# still covering headline + summary + current-role description.
EMBED_MAX_SEQ_LENGTH = 256
# bge retrieval is asymmetric: this instruction is prepended to the QUERY side
# (JD aspect queries) only, never to candidate "passages". See bge model card.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# --- artifact filenames (offline -> online handoff; ARCHITECTURE.md §4) -----
CANDIDATE_EMB_FILE = "candidate_emb.npy"     # float32 (N, EMBED_DIM), L2-normalized
CANDIDATE_IDS_FILE = "candidate_ids.npy"     # str array, length N, candidate file order
FAISS_INDEX_FILE = "faiss.index"             # FAISS IndexFlatIP over the embeddings
JD_PROFILE_FILE = "jd_profile.json"          # JD requirement profile (jd_profile.py)

# --- cross-encoder reranker (issue #2) -------------------------------------
# Downloaded ONCE offline by scripts/download_cross_encoder.py into
# CROSS_ENCODER_MODEL_DIR; rank.py loads only from that local dir (the saved
# bytes are the determinism pin — no revision lookup, no network at rank time).
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_MODEL_DIR = MODELS_DIR / "ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_BATCH_SIZE = 32     # CPU-friendly pair batch size for rerank
# Truncate each (JD, candidate) pair to this many tokens. The model was trained
# on <=512-token passages; candidate texts run 200-740 tokens, so without this
# cap we feed out-of-distribution sequences AND pay quadratic attention cost on
# the wasted tail. MUST be set — None lets sequences exceed 512 and warns of
# indexing errors.
CROSS_ENCODER_MAX_LENGTH = 256
# The reranker only needs to SHARPEN the head (top ~50-100, where NDCG@10/@50
# lives), so rerank a modest shortlist, not all ~800. Uncapped this stage alone
# blew the 5-min budget; this + the max-length cap keep it well inside it.
CROSS_ENCODER_SHORTLIST_SIZE = 200
