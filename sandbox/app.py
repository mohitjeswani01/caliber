"""Sandbox demo app (Stage-1 requirement: hosted small-sample reproducibility).

A minimal hosted entry point (e.g. Streamlit / HF Spaces) that accepts a small
candidate sample (≤100 records), runs the full Caliber ranking pipeline
end-to-end on CPU within the ≤5-min budget, and returns the ranked CSV.

It does NOT need to handle the full 100K pool — small-sample reproducibility is
all the sandbox verifies (submission_spec §10.5). It reuses the same
``src/caliber`` code path as ``rank.py`` so what reviewers run matches what we
submit. Stub only — no logic yet.
"""

if __name__ == "__main__":
    raise SystemExit("sandbox/app.py is a stub — not implemented yet.")
