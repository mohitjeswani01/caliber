"""Grounded, template-based reasoning generation (ONLINE, no LLM).

Produces the 1–2 sentence ``reasoning`` string for each ranked candidate,
deterministically from extracted profile facts — never invented. Per the Stage-4
manual-review checks, each entry must:

- cite specific facts (years, current title, named skills, signal values),
- connect to a specific JD requirement,
- honestly acknowledge real gaps/concerns (notice period, services-only, etc.),
- vary meaningfully between candidates, and
- match its tone to the rank (no glowing rank-95, no critical rank-5).

Pure templating over real fields is deliberately safer than an LLM here: it
*cannot* hallucinate a skill or employer the candidate doesn't have, which is
exactly what the review penalises.
"""
