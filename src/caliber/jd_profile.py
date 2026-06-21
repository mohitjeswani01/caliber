"""The structured JD requirement profile — its structure, intent, and contents.

The machine-readable backbone the online scorer reasons against is
``artifacts/jd_profile.json``. It captures:

- must-haves (embeddings-based retrieval, vector/hybrid search, eval frameworks,
  strong Python), nice-to-haves (LoRA/PEFT, learning-to-rank, HR-tech, OSS),
- explicit disqualifiers (pure research, <12mo LangChain-only, no code in 18mo,
  title-chaser, services-only career, CV/speech/robotics without NLP/IR),
- location preferences (India Tier-1 / relocation-willing) and the 5–9yr band,
- the canonical query text(s) we embed to score semantic similarity.

The profile is hand-curated for the target role and committed directly as
``artifacts/jd_profile.json`` (a small, static artifact), then loaded at runtime
via ``scorer.load_jd_profile_artifact``. THIS module documents the profile's
structure and intent; the committed artifact is the source of truth. The
artifact is static and LLM-free at runtime.
"""
