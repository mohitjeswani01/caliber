"""OFFLINE precompute — build all artifacts ``rank.py`` depends on.

No time limit; runs once on the full 100K pool and persists to ``artifacts/``:

    1. parse the JD → ``jd_profile.json`` (jd_profile)
    2. build rich per-candidate text + encode → ``candidate_embeddings.npy``
       (embeddings)
    3. build the FAISS index → ``faiss.index`` (+ id mapping) (index)
    4. fit/persist BM25 state over role descriptions
    5. record precompute wall-clock into ``submission_metadata.yaml``

This is allowed to be slow and may use an LLM for offline label work elsewhere,
but everything it emits is static so the online ranker stays CPU-only and
network-free. Stub only — no logic yet.
"""

if __name__ == "__main__":
    raise SystemExit("precompute.py is a stub — not implemented yet.")
