"""Memory-safe I/O for the 100K pool and submission CSV writing.

The candidate file is ~465 MB uncompressed, so we never naively load it all.
Responsibilities:

- Stream ``candidates.jsonl`` (and ``.jsonl.gz``) line-by-line, yielding parsed
  records, so peak memory stays well under the 16 GB budget.
- Load/save the offline artifacts (embeddings ``.npy``, FAISS index, JD profile
  JSON, BM25 state) from ``artifacts/``.
- Write the final submission CSV in exact spec order
  (``candidate_id,rank,score,reasoning``), UTF-8, correctly quoted, and run it
  through the bundled validator before declaring success.

Pure plumbing — no ranking decisions.
"""
