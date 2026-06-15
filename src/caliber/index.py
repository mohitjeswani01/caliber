"""FAISS index build (OFFLINE) and semantic search (ONLINE).

Persists a FAISS index over the candidate embeddings to ``artifacts/`` and
provides fast cosine/IP similarity lookups online.

Responsibilities:
- Build the index from ``candidate_embeddings.npy`` with a row-order ⇄
  candidate_id mapping so results join back deterministically.
- Online: given the JD query vector(s), return per-candidate semantic similarity
  scores for the full pool (or a generous candidate set) within the CPU budget.

Pure retrieval. Note: semantic similarity is ONE input to the hybrid score —
never the sole ranking signal (raw cosine is the losing baseline).
"""
