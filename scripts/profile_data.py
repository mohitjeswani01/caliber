"""OFFLINE data profiling — understand the pool before modelling.

Streams ``candidates.jsonl`` and reports the distributions STRATEGY.md relies on
so our assumptions are grounded in the actual data, not the spec's prose:

- title frequencies (ML/AI titles vs noise vs adjacent hidden-gem titles),
- experience-band histogram, geography breakdown,
- redrob_signals envelopes (response-rate median, open_to_work %, github -1 %),
- candidate field coverage and obvious honeypot-shaped anomalies.

Read-only analysis; writes summaries to ``docs/`` for our own reference.
Stub only — no logic yet.
"""

if __name__ == "__main__":
    raise SystemExit("profile_data.py is a stub — not implemented yet.")
