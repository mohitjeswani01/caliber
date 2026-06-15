"""Structured relevance features with skill-gating (ONLINE).

Computes the Section-4 structured signals from extracted profile facts — the
substance backbone of the score. Each candidate yields a feature vector:

- role substance (career history shows retrieval/ranking/recommender/applied-ML
  at product companies) — the dominant signal,
- experience-band fit (5–9yr, ideal 6–8), NLP/IR vs CV/speech/robotics,
- product-company vs services-only career, recent-shipping vs architecture-only,
- title-chaser tenure check, external validation (OSS/GitHub), location fit.

The critical mechanism is **skill-gating**: a listed skill earns credit only
when the role history corroborates it. This is the primary defense against
keyword stuffers — it severs "lists RAG" from "did RAG".
"""
