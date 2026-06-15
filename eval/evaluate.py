"""OFFLINE harness — score a ranking against silver labels.

Glue between a produced ranking and ``eval/metrics.py``: loads the silver-label
set, takes a candidate ranking (or runs the ranker to produce one), computes the
composite + each component, and prints a report used to tune weights.

Also runs the automated sanity checks from STRATEGY.md §7:
- no honeypots in our top 100 (and ideally none near the top),
- no non-tech keyword-stuffer in the top 50,
- known plain-language fits actually surface.

This is how we know we are winning before we spend one of our 3 submissions.
Stub only — no logic yet.
"""

if __name__ == "__main__":
    raise SystemExit("evaluate.py is a stub — not implemented yet.")
