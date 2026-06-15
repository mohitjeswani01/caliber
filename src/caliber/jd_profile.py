"""Build the structured JD requirement profile (OFFLINE).

Turns the free-text job description into ``artifacts/jd_profile.json`` — the
machine-readable backbone the online scorer reasons against. Captures:

- must-haves (embeddings-based retrieval, vector/hybrid search, eval frameworks,
  strong Python), nice-to-haves (LoRA/PEFT, learning-to-rank, HR-tech, OSS),
- explicit disqualifiers (pure research, <12mo LangChain-only, no code in 18mo,
  title-chaser, services-only career, CV/speech/robotics without NLP/IR),
- location preferences (India Tier-1 / relocation-willing) and the 5–9yr band,
- the canonical query text(s) we embed to score semantic similarity.

This runs offline with no time limit; an LLM may *assist* in drafting the
profile, but the persisted artifact is static and LLM-free at runtime.
"""
