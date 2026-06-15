"""Local sentence-transformer encoding (OFFLINE) + query encoding (ONLINE).

Owns the single embedding model used in both phases (e.g.
``BAAI/bge-small-en-v1.5`` or ``intfloat/e5-small-v2``), loaded from a
locally-cached path so neither phase hits the network.

Responsibilities:
- Build the rich per-candidate text representation (headline + summary + role
  titles + role **descriptions** + skills-with-context) — descriptions matter
  because plain-language Tier-5s never use the buzzwords.
- Encode all 100K candidates to ``candidate_embeddings.npy`` (offline, batched).
- Encode the JD query text(s) online (a handful of vectors — trivially cheap).

Encoding the *descriptions*, not the skill tags, is what surfaces hidden gems.
"""
