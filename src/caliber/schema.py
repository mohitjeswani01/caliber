"""Typed access to a candidate record + schema-aware field extraction.

Wraps the raw ``candidates.jsonl`` record (shape defined in
``data/challenge/candidate_schema.json``) so the rest of the pipeline never
reaches into raw dicts. Responsibilities:

- Parse a single JSON line into a lightweight, attribute-accessible candidate
  object (profile, career_history, education, skills, redrob_signals).
- Provide safe accessors that tolerate missing/optional fields (certifications,
  languages, null end_dates) without raising.
- Normalise enums (company_size bands, proficiency, work_mode) into comparable
  forms used by ``features`` and ``honeypot``.

No scoring here — extraction and normalisation only.
"""
