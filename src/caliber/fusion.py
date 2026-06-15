"""Combine score components into one base relevance score (ONLINE).

Fuses the heterogeneous signals — semantic similarity, lexical BM25, and the
gated structured features — into a single base relevance score per candidate,
per the STRATEGY.md hybrid formula:

    base = semantic + lexical(BM25) + gated_structured_features

Handles per-component normalisation so no single signal dominates by scale, and
exposes the linear weights (from ``config``) for tuning against silver labels.
Either this linear fusion OR the LTR model produces the base score that
``scorer`` then multiplies by the behavioral multiplier — they are the two
interchangeable backbones we compare.
"""
