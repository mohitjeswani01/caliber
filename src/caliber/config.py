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

# --- determinism -----------------------------------------------------------
SEED = 42

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
# bge retrieval is asymmetric: the instruction is prepended to the QUERY side
# (the JD aspect queries) only, never to the candidate "passages". See bge model
# card. Applied in embeddings.encode_texts(is_query=True).
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# --- artifact filenames (offline -> online handoff; ARCHITECTURE.md §4) -----
CANDIDATE_EMB_FILE = "candidate_emb.npy"     # float32 (N, EMBED_DIM), L2-normalized
CANDIDATE_IDS_FILE = "candidate_ids.npy"     # str array, length N, candidate file order
FAISS_INDEX_FILE = "faiss.index"             # FAISS IndexFlatIP over the embeddings
JD_PROFILE_FILE = "jd_profile.json"          # JD requirement profile (jd_profile.py)
