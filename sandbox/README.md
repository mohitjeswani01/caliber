---
title: Caliber
emoji: 🎯
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
short_description: Ranking AI-engineer candidates by fit, not keywords.
---

# Caliber — fit, not keywords

A live demo of **Caliber**, our entry to the Redrob *India Runs · Track 1*
Intelligent Candidate Discovery & Ranking Challenge. It runs the **real**
`src/caliber` ranking pipeline (semantic retrieval + BM25 + gated structured
features + cross-encoder rerank + behavioural multiplier + honeypot floor) over a
curated **150-candidate** pool — CPU-only, fully deterministic — and shows the
ranked top fits **and** the impostors it rejects: keyword-stuffers gated to the
floor and internally-impossible honeypots flagged.
