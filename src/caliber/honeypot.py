"""Honeypot / internal-consistency detector (ONLINE).

Flags the ~80 deliberately-impossible profiles so they can be forced to the
score floor. Detection is by internal contradiction, not keywords:

- role/skill tenure exceeding the candidate's total experience or the company's
  plausible age (e.g. 8 yrs at a 3-yr-old company),
- "expert" proficiency in many skills with 0 months used,
- summed role durations inconsistent with years_of_experience,
- impossible date ranges (end before start, future dates).

Returns a per-candidate boolean + the triggering reason (for reasoning/audit).
Ranking honeypots in the top 100 is a Stage-3 disqualifier (>10% kills us), so
this detector is precision-tuned: a false positive on a real fit is costly too.
"""
