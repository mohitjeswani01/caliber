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

from pathlib import Path

# --- Filesystem layout -------------------------------------------------------
# Repo root is two levels up from this file: src/caliber/config.py -> repo root.
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODELS_DIR = ROOT_DIR / "models"  # gitignored; populated offline (precompute)

# --- Determinism -------------------------------------------------------------
SEED = 42                         # fixed everywhere (ARCHITECTURE.md §5)
TIE_BREAK_KEY = "candidate_id"    # final tie-break, ascending

# --- Models (cached locally so the ONLINE path never hits the network) -------
# The embedding model is already cached at models/bge-small-en-v1.5 (saved
# SentenceTransformer dir, loadable by path). The constant is recorded here so
# every consumer agrees on the same on-disk location.
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_MODEL_DIR = MODELS_DIR / "bge-small-en-v1.5"

# Cross-encoder reranker (issue #2). Downloaded ONCE offline by
# scripts/download_cross_encoder.py into CROSS_ENCODER_MODEL_DIR; rank.py loads
# only from that local dir (the saved bytes are the determinism pin — no
# revision lookup, no network at rank time).
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_MODEL_DIR = MODELS_DIR / "ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_BATCH_SIZE = 32     # CPU-friendly pair batch size for rerank
# Truncate each (JD, candidate) pair to this many tokens. The model was trained
# on <=512-token passages; candidate texts run 200-740 tokens, so without this
# cap we feed out-of-distribution sequences AND pay quadratic attention cost on
# the wasted tail. 256 keeps the front-loaded signal (headline + summary +
# recent roles) while roughly halving CPU cost vs 512. MUST be set — leaving it
# None lets sequences exceed 512 and warns of indexing errors.
CROSS_ENCODER_MAX_LENGTH = 256
# The cross-encoder only needs to SHARPEN the head of the list (top ~50-100 is
# where NDCG@10/@50 lives), so rerank a modest shortlist, not all ~800. At
# ~1.1s/pair uncapped this stage alone blew the 5-min budget; this cap + the
# max-length cap keep it well inside it. Tune against silver labels.
CROSS_ENCODER_SHORTLIST_SIZE = 200
