"""Throughput + quality check for cross_encoder.rerank (sandbox).

Run with `python -u` so prints flush as they happen.
"""
import json
import sys
import time

sys.path.insert(0, "src")
import numpy as np

from caliber import cross_encoder, config

data = json.load(open("data/challenge/sample_candidates.json"))


def to_text(c):
    p = c.get("profile", {})
    parts = [p.get("headline") or "", p.get("summary") or ""]
    for r in c.get("career_history", []):
        title = r.get("title") or ""
        company = r.get("company") or ""
        desc = r.get("description") or ""
        parts.append(f"{title} at {company}. {desc}")
    sk = ", ".join(s.get("name", "") for s in c.get("skills", []))
    if sk:
        parts.append("Skills: " + sk)
    return "\n".join(x for x in parts if x.strip())


texts = [to_text(c) for c in data]
m = cross_encoder.load_cross_encoder()
print("configured max_length:", m.max_length)

jd = ("Senior AI Engineer with embeddings, retrieval, ranking, recommendation, "
      "LLMs, fine-tuning; shipped production ML at a product company; 5-9 years; "
      "NLP/IR background.")

# Quality: strong fit must still outrank stuffers after truncation.
byid = {c["candidate_id"]: t for c, t in zip(data, texts)}
ids = ["CAND_0000031", "CAND_0000033", "CAND_0000024", "CAND_0000026"]
qs = cross_encoder.rerank(jd, [byid[i] for i in ids])
print("quality (truncated) strong vs stuffers:",
      [(i, round(s, 3)) for i, s in zip(ids, qs)])

cross_encoder.rerank(jd, texts[:8])  # warm
for N in (200, 400):
    subset = [texts[i % len(texts)] for i in range(N)]
    t0 = time.perf_counter()
    cross_encoder.rerank(jd, subset)
    dt = time.perf_counter() - t0
    print(f"N={N}: {dt:.2f}s total, {dt / N * 1000:.1f} ms/pair, "
          f"throughput {N / dt:.1f} pairs/s")
