"""Behavioral multiplier from the 23 redrob_signals (ONLINE).

Converts platform engagement signals into a single bounded multiplier
(e.g. 0.5–1.15) applied on top of the base substance score, so a
strong-on-paper but unavailable candidate is pushed down without being erased.

Positive: recent last_active_date, open_to_work_flag, high
recruiter_response_rate, saved_by_recruiters_30d, interview_completion_rate,
profile_completeness_score, verified email/phone, healthy github_activity_score.
Negative: stale last-active, very low response rate, low interview completion,
very long notice_period_days.

Uses normalised envelopes (not raw values) and stays bounded so behaviour
*modulates* rather than dominates substance. This is what separates behavioral
twins — identical profiles, different availability.
"""
