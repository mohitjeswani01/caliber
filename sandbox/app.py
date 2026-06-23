"""Caliber — premium hosted demo (Gradio / HuggingFace Spaces).

Runs the REAL Caliber ranking pipeline on a curated 150-candidate pool and renders
the result as a Redrob-branded product page. It is NOT a re-implementation: it calls
the exact same ``src/caliber`` functions ``rank.py`` calls
(``score_candidates`` -> ``build_submission_rows`` -> ``reasoning_for``), pointed at
the tiny artifacts shipped in ``demo_data/`` instead of the 100K pool.

WHY IT RUNS BOTH LOCALLY AND ON A SPACE
---------------------------------------
* Importing ``caliber``:
    - On the Space, ``requirements.txt`` installs the package straight from the
      public GitHub repo (``caliber @ git+https://github.com/...``), so ``import
      caliber`` resolves to the real package — no editable install needed.
    - Locally (repo checkout), the package is already importable (``pip install -e
      .``). If it is somehow NOT installed, we fall back to putting the repo's
      ``src/`` on ``sys.path``. The SAME app.py therefore works in every case.
* Resolving the demo artifacts:
    - Every path is computed RELATIVE TO ``__file__`` (this file), never absolute.
      ``demo_data/`` ships inside the Space repo next to ``app.py``, and sits next
      to it in the local checkout too — so the same code finds it in any working
      directory, local or hosted.

MODEL DOWNLOAD
--------------
The hosted demo is allowed to download the two small CPU models (bge-small +
ms-marco cross-encoder) on first run — we set ``CALIBER_ALLOW_MODEL_DOWNLOAD=1``
*before* importing caliber. (The zero-network rule is a constraint on the JUDGED
``rank.py`` path only, never this hosted demo.) The pipeline result is computed once
and memoised in module scope, so repeated "Rank" clicks are instant.
"""

from __future__ import annotations

import html
import os
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------- #
# 1. Environment — MUST be set before caliber.embeddings.load_model is reached.
#    This hosted demo opts IN to downloading the small models once. We must NOT
#    import rank.py here: it hard-locks HF_HUB_OFFLINE=1 at module import (correct
#    for the judged path) which would forbid that download on the Space. We instead
#    call the same underlying caliber loaders rank.py uses.
# --------------------------------------------------------------------------- #
os.environ.setdefault("CALIBER_ALLOW_MODEL_DOWNLOAD", "1")

# --------------------------------------------------------------------------- #
# 2. Import the REAL package — resilient across environments (see module docstring).
# --------------------------------------------------------------------------- #
try:
    import caliber  # noqa: F401  (installed: -e locally OR git+https on the Space)
except ImportError:  # local repo without an install — fall back to src/ on path
    _repo_src = Path(__file__).resolve().parents[1] / "src"
    if _repo_src.exists():
        sys.path.insert(0, str(_repo_src))
    import caliber  # noqa: F401

import numpy as np
import gradio as gr

from caliber import config, ranker
from caliber.index import load_index
from caliber.io_utils import stream_candidates
from caliber.reasoning import reasoning_for
from caliber.scorer import load_jd_profile_artifact, score_candidates

# --------------------------------------------------------------------------- #
# 3. Paths — resolved RELATIVE TO this file so they hold local AND on the Space.
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
DEMO_DIR = HERE / "demo_data"
ARTIFACTS_DIR = DEMO_DIR / "artifacts"
CANDIDATES_PATH = DEMO_DIR / "demo_candidates.jsonl"

TOP_CARDS = 15            # how many ranked candidates to render as hero cards
STUFFER_GATE = 0.5        # skill_corroboration < this == uncorroborated AI skills
                          # (the exact threshold reasoning.py uses for the
                          #  "keyword-stuffing signal" clause)
# The CSV submission caps reasoning at 320 chars (ranker default), which lands
# mid-sentence and shows a trailing "…" on a card. For the richer card UI we ask
# the SAME grounded reasoning_for for the full, un-truncated string (it tops out
# ~470 chars and always ends on a clean clause/period). Same function, same facts —
# only the display length differs; the submission path is untouched.
CARD_REASONING_LEN = 600

# --------------------------------------------------------------------------- #
# Design tokens (Redrob palette) — kept in Python so the CSS below stays DRY.
# --------------------------------------------------------------------------- #
TOKENS = {
    "bg": "#08080c", "surface": "#121219", "surface2": "#1a1a24",
    "border": "#262633", "text": "#f4f4f8", "text_dim": "#9a9aae",
    "accent_a": "#7c3aed", "accent_b": "#2563eb",
    "flag": "#f59e0b", "floor": "#ef4444", "good": "#34d399",
}


# --------------------------------------------------------------------------- #
# Pipeline — call the EXACT functions rank.py calls. Loaded/computed once.
# --------------------------------------------------------------------------- #
def _load_artifacts():
    """Load (jd_profile, candidate_ids, faiss_index, candidates_by_id) from the demo
    folder using the real caliber loaders — the same handoff rank.py performs
    (load_jd_profile_artifact / candidate_ids.npy / index.load_index /
    io_utils.stream_candidates). Cheap; no model needed yet."""
    jd_profile = load_jd_profile_artifact(ARTIFACTS_DIR / config.JD_PROFILE_FILE)
    ids = np.load(ARTIFACTS_DIR / config.CANDIDATE_IDS_FILE, allow_pickle=True)
    candidate_ids = [str(x) for x in ids.tolist()]
    faiss_index = load_index(str(ARTIFACTS_DIR / config.FAISS_INDEX_FILE))
    candidates_by_id = {
        str(c.candidate_id): c for c in stream_candidates(CANDIDATES_PATH)
    }
    return jd_profile, candidate_ids, faiss_index, candidates_by_id


def _ensure_cross_encoder_source() -> str:
    """Make the cross-encoder resolvable in BOTH environments.

    ``caliber.cross_encoder.load_cross_encoder`` always loads from the local dir
    ``config.CROSS_ENCODER_MODEL_DIR`` — unlike ``embeddings.load_model`` it has NO
    hub fallback. That local dir is shipped on a dev box but NOT on an HF Space, so
    the Space crashes with ``OSError: Can't load the configuration of …/models/
    ms-marco-MiniLM-L-6-v2``.

    We fix it from here WITHOUT touching the installed package: if the local dir is
    missing (the Space), we repoint ``config.CROSS_ENCODER_MODEL_DIR`` to the HUB
    REPO ID (``config.CROSS_ENCODER_MODEL_NAME`` = "cross-encoder/ms-marco-MiniLM-
    L-6-v2"). With ``CALIBER_ALLOW_MODEL_DOWNLOAD=1`` already set, the loader skips
    its offline lock and hands that id straight to ``CrossEncoder``, which downloads
    + caches it from the hub — exactly how the bge embeddings model already loads on
    the Space. If the local dir EXISTS (a local run, where models/ is present), we
    leave it untouched so the real local path keeps using the cached model offline.

    ``cross_encoder.load_cross_encoder`` reads ``config.CROSS_ENCODER_MODEL_DIR`` at
    call time and ``from . import config`` is the same module object we mutate here,
    so the override is picked up by the first (and only) load. Returns a short source
    description for the startup log.
    """
    local = config.CROSS_ENCODER_MODEL_DIR
    has_local = (
        isinstance(local, Path) and local.is_dir() and (local / "config.json").exists()
    )
    if has_local:
        return f"local dir ({local})"
    # Absent (HF Space): resolve to the hub repo id (a plain string). On the
    # download-allowed path the loader passes it directly to CrossEncoder.
    config.CROSS_ENCODER_MODEL_DIR = config.CROSS_ENCODER_MODEL_NAME
    return f"HuggingFace hub ('{config.CROSS_ENCODER_MODEL_NAME}', downloaded once)"


_CE_SOURCE = _ensure_cross_encoder_source()
print(f"[app] cross-encoder source: {_CE_SOURCE}")

# Load the static artifacts at import (fast). The model download + scoring is
# deferred to the first compute() call and then cached.
_JD, _IDS, _INDEX, _CANDS = _load_artifacts()

_RESULT_CACHE: dict | None = None   # memoised pipeline output (computed once)


def compute_ranking() -> dict:
    """Run the real pipeline ONCE and memoise it.

    Calls ``score_candidates`` (semantic retrieval + BM25 + structured features +
    honeypot detection + cross-encoder rerank + behavioural multiplier + floor) and
    ``build_submission_rows`` (top-100 selection + invariants + grounded reasoning),
    exactly as ``rank.produce_submission`` does. Returns a structured dict the
    renderer turns into HTML. First call downloads the two small models and takes
    ~30-60s on CPU; every later call returns the cached result instantly.
    """
    global _RESULT_CACHE
    if _RESULT_CACHE is not None:
        return _RESULT_CACHE

    t0 = time.perf_counter()
    results = score_candidates(
        jd_profile=_JD,
        candidate_ids=_IDS,
        faiss_index=_INDEX,
        candidates_by_id=_CANDS,
        ce_enabled=True,
    )
    rows = ranker.build_submission_rows(results, top_n=ranker.TOP_N, candidates=_CANDS)
    elapsed = time.perf_counter() - t0

    # --- top ranked candidates (hero cards) ---
    top_score = max((r.score for r in rows), default=1.0) or 1.0
    top = []
    for r in rows[:TOP_CARDS]:
        c = _CANDS[r.candidate_id]
        p = c.profile
        top.append({
            "rank": r.rank,
            "name": p.anonymized_name,
            "title": p.current_title,
            "company": p.current_company,
            "years": p.years_of_experience,
            "score": r.score,
            "pct": max(8.0, min(100.0, r.score / top_score * 100.0)),
            # Full grounded reasoning for the card (clean sentence, no mid-word "…").
            "reasoning": reasoning_for(results[r.candidate_id], c, max_len=CARD_REASONING_LEN),
        })

    # --- the rejects: honeypots floored + keyword-stuffers gated ---
    honeypots, stuffers = [], []
    for cs in results.values():
        c = _CANDS.get(cs.candidate_id)
        if c is None:
            continue
        p = c.profile
        if cs.is_honeypot:
            honeypots.append({
                "name": p.anonymized_name,
                "title": p.current_title,
                "company": p.current_company,
                "reason": "; ".join(cs.honeypot_reasons) or "internal inconsistency",
            })
        elif cs.feature_dict.get("skill_corroboration", 1.0) < STUFFER_GATE:
            stuffers.append({
                "name": p.anonymized_name,
                "title": p.current_title,
                "company": p.current_company,
                "n_skills": len(c.skills),
                "reason": (
                    f"{len(c.skills)} AI skills claimed, none corroborated by the "
                    "career history — gated to the score floor"
                ),
            })
    honeypots.sort(key=lambda d: d["name"])
    stuffers.sort(key=lambda d: -d["n_skills"])

    _RESULT_CACHE = {
        "n_pool": len(_CANDS),
        "elapsed": elapsed,
        "n_stuffers": len(stuffers),
        "n_honeypots": len(honeypots),
        "top": top,
        "honeypots": honeypots,
        "stuffers": stuffers,
    }
    return _RESULT_CACHE


# --------------------------------------------------------------------------- #
# Rendering — custom HTML using the Redrob design tokens.
# --------------------------------------------------------------------------- #
def _css() -> str:
    t = TOKENS
    return f"""
<style>
  .cal {{
    --bg:{t['bg']}; --surface:{t['surface']}; --surface-2:{t['surface2']};
    --border:{t['border']}; --text:{t['text']}; --text-dim:{t['text_dim']};
    --accent-a:{t['accent_a']}; --accent-b:{t['accent_b']};
    --flag:{t['flag']}; --floor:{t['floor']}; --good:{t['good']};
    font-family: system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
    color: var(--text); max-width: 1080px; margin: 0 auto; padding: 4px 18px 64px;
  }}
  .cal * {{ box-sizing: border-box; }}

  /* ---------- hero ---------- */
  .cal .hero {{ position: relative; text-align: center; padding: 54px 16px 30px; overflow: hidden; }}
  .cal .glow {{
    position: absolute; inset: -40% 0 auto 0; height: 360px; pointer-events: none;
    background:
      radial-gradient(620px 280px at 38% 18%, rgba(124,58,237,.30), transparent 70%),
      radial-gradient(620px 280px at 66% 30%, rgba(37,99,235,.26), transparent 70%),
      radial-gradient(420px 200px at 50% 0%, rgba(245,158,11,.06), transparent 70%);
    filter: blur(8px); z-index: 0;
  }}
  .cal .hero > * {{ position: relative; z-index: 1; }}
  .cal h1 {{
    font-family: Georgia,'Times New Roman',serif; font-style: italic;
    font-size: 84px; line-height: .96; margin: 0; letter-spacing: -.02em;
    background: linear-gradient(92deg, var(--text) 30%, var(--accent-a) 70%, var(--accent-b));
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
  }}
  .cal .tag {{ color: var(--text-dim); font-size: 19px; margin: 14px 0 0; }}
  .cal .tag em {{ color: var(--text); font-family: Georgia,serif; font-style: italic; }}
  .cal .kicker {{
    display:inline-block; margin-bottom: 18px; padding: 5px 14px; border-radius: 999px;
    border: 1px solid var(--border); background: var(--surface);
    color: var(--text-dim); font-size: 12px; letter-spacing: .18em; text-transform: uppercase;
  }}

  /* ---------- stats strip ---------- */
  .cal .stats {{
    display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; margin: 26px 0 8px;
  }}
  .cal .stat {{
    background: var(--surface); border: 1px solid var(--border); border-radius: 16px;
    padding: 22px 18px; text-align: center; box-shadow: 0 8px 30px rgba(0,0,0,.45);
  }}
  .cal .stat .num {{
    display: block; font-size: 38px; font-weight: 800; letter-spacing: -.02em;
    background: linear-gradient(90deg, var(--accent-a), var(--accent-b));
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
  }}
  .cal .stat.amber .num {{ background: linear-gradient(90deg,#fbbf24,var(--flag)); -webkit-background-clip:text; background-clip:text; }}
  .cal .stat.red .num   {{ background: linear-gradient(90deg,#fb7185,var(--floor)); -webkit-background-clip:text; background-clip:text; }}
  .cal .stat .lbl {{ display:block; margin-top: 6px; color: var(--text-dim); font-size: 13px; }}

  /* ---------- JD panel ---------- */
  .cal .jd {{
    background: linear-gradient(180deg, var(--surface), var(--bg));
    border: 1px solid var(--border); border-radius: 16px; padding: 24px 26px; margin: 26px 0;
  }}
  .cal .jd h2 {{ margin: 0 0 4px; font-size: 13px; letter-spacing: .16em; text-transform: uppercase; color: var(--text-dim); }}
  .cal .jd .role {{ font-family: Georgia,serif; font-style: italic; font-size: 30px; margin: 2px 0 4px; }}
  .cal .jd .band {{ color: var(--text-dim); font-size: 14px; margin-bottom: 16px; }}
  .cal .jd .aspects {{ display: grid; grid-template-columns: repeat(2,1fr); gap: 10px; }}
  .cal .aspect {{ background: var(--surface-2); border: 1px solid var(--border); border-radius: 12px; padding: 12px 14px; }}
  .cal .aspect .top {{ display:flex; justify-content: space-between; align-items:center; gap: 8px; }}
  .cal .aspect .nm {{ font-weight: 700; font-size: 14px; }}
  .cal .aspect .w {{ font-size: 12px; color: var(--accent-b); font-weight: 700; }}
  .cal .aspect .wbar {{ height: 4px; border-radius: 999px; background: var(--border); margin: 8px 0; overflow: hidden; }}
  .cal .aspect .wbar > span {{ display:block; height:100%; background: linear-gradient(90deg,var(--accent-a),var(--accent-b)); }}
  .cal .aspect .qt {{ color: var(--text-dim); font-size: 12px; line-height: 1.5; margin: 0; }}

  /* ---------- section titles ---------- */
  .cal .sec {{ display:flex; align-items: baseline; gap: 12px; margin: 40px 0 16px; }}
  .cal .sec h2 {{ font-family: Georgia,serif; font-style: italic; font-size: 30px; margin: 0; }}
  .cal .sec .hint {{ color: var(--text-dim); font-size: 14px; }}
  .cal .caught-title h2 {{ color: #fda4af; }}

  /* ---------- ranked cards ---------- */
  .cal .cards {{ display: flex; flex-direction: column; gap: 12px; }}
  .cal .card {{
    position: relative; display: grid; grid-template-columns: 64px 1fr; gap: 18px;
    background: var(--surface); border: 1px solid var(--border); border-radius: 16px;
    padding: 18px 20px 18px 16px; box-shadow: 0 8px 30px rgba(0,0,0,.45);
    transition: transform .12s ease, border-color .12s ease;
  }}
  .cal .card:hover {{ transform: translateY(-2px); border-color: #34344a; }}
  .cal .rank {{
    align-self: start; font-family: Georgia,serif; font-style: italic; font-weight: 700;
    font-size: 30px; text-align: center; padding-top: 4px;
    background: linear-gradient(180deg, var(--accent-a), var(--accent-b));
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
  }}
  .cal .who h3 {{ margin: 0; font-size: 19px; font-weight: 700; }}
  .cal .who p {{ margin: 3px 0 0; color: var(--text-dim); font-size: 14px; }}
  .cal .barrow {{ display:flex; align-items:center; gap: 12px; margin: 12px 0 10px; }}
  .cal .bar {{ flex: 1; height: 9px; border-radius: 999px; background: var(--surface-2); overflow: hidden; }}
  .cal .bar > span {{ display:block; height:100%; border-radius: 999px;
    background: linear-gradient(90deg, var(--accent-a), var(--accent-b)); }}
  .cal .score {{ font-variant-numeric: tabular-nums; font-weight: 700; font-size: 14px; color: var(--text); min-width: 52px; text-align: right; }}
  .cal .reason {{ margin: 0; color: #cdcdda; font-size: 14px; line-height: 1.55; }}

  /* ---------- caught / rejected cards ---------- */
  .cal .caught {{ display: grid; grid-template-columns: repeat(2,1fr); gap: 12px; }}
  .cal .card.flagged {{ display:block; padding: 16px 18px; border-left: 4px solid var(--flag); }}
  .cal .card.floored {{ display:block; padding: 16px 18px; border-left: 4px solid var(--floor); }}
  .cal .badge {{
    display:inline-block; font-size: 11px; font-weight: 800; letter-spacing: .12em;
    padding: 4px 11px; border-radius: 999px; margin-bottom: 10px;
  }}
  .cal .badge.flag  {{ color: #1a1206; background: var(--flag); }}
  .cal .badge.floor {{ color: #fff; background: var(--floor); }}
  .cal .card.flagged h3, .cal .card.floored h3 {{ margin: 0; font-size: 17px; }}
  .cal .card.flagged p.sub, .cal .card.floored p.sub {{ margin: 2px 0 8px; color: var(--text-dim); font-size: 13px; }}
  .cal .card.flagged .reason, .cal .card.floored .reason {{ font-size: 13px; }}
  .cal .card.floored .reason {{ color: #fca5a5; }}
  .cal .card.flagged .reason {{ color: #fcd34d; }}

  .cal .foot {{ text-align:center; color: var(--text-dim); font-size: 13px; margin-top: 44px;
    border-top: 1px solid var(--border); padding-top: 22px; }}
  .cal .foot b {{ color: var(--text); }}

  /* ---------- landing call-to-action hint ---------- */
  .cal .cta-hint {{ text-align: center; color: var(--text-dim); font-size: 15px;
    line-height: 1.6; max-width: 620px; margin: 26px auto 4px; }}
  .cal .cta-hint b {{ color: var(--text); }}

  @media (max-width: 720px) {{
    .cal h1 {{ font-size: 56px; }}
    .cal .stats, .cal .jd .aspects, .cal .caught {{ grid-template-columns: 1fr; }}
  }}
</style>
"""


def _esc(x) -> str:
    return html.escape(str(x))


def _fmt_years(y) -> str:
    try:
        return f"{float(y):.1f}".rstrip("0").rstrip(".") + " yrs"
    except (TypeError, ValueError):
        return "—"


def render_jd() -> str:
    """The fixed Senior AI Engineer JD, read from the demo jd_profile.json."""
    role = _esc(_JD.get("role", "Senior AI Engineer"))
    band = _JD.get("experience_band", {}) or {}
    band_txt = ""
    if band:
        band_txt = (f"Target experience {band.get('min','?')}–{band.get('max','?')} yrs "
                    f"(ideal {band.get('ideal_min','?')}–{band.get('ideal_max','?')})")
    aspects = sorted((_JD.get("aspects", {}) or {}).items(),
                     key=lambda kv: -float(kv[1].get("weight", 0)))
    cards = []
    for name, a in aspects:
        w = float(a.get("weight", 0))
        qt = (a.get("query_text", "") or "")
        qt = qt[:150].rsplit(" ", 1)[0] + "…" if len(qt) > 150 else qt
        cards.append(f"""
          <div class="aspect">
            <div class="top"><span class="nm">{_esc(name.replace('_',' ').title())}</span>
              <span class="w">{w*100:.0f}%</span></div>
            <div class="wbar"><span style="width:{w*100:.0f}%"></span></div>
            <p class="qt">{_esc(qt)}</p>
          </div>""")
    return f"""
      <div class="jd">
        <h2>The role we rank against</h2>
        <div class="role">{role}</div>
        <div class="band">{_esc(band_txt)} · weighted requirement aspects below</div>
        <div class="aspects">{''.join(cards)}</div>
      </div>"""


def render_results_body(data: dict) -> str:
    """Stats strip + JD panel + ranked cards + 'caught these' — the post-click view
    (no outer .cal wrapper / hero; the caller adds those)."""
    secs = f"{data['elapsed']:.1f}"

    # The stat counts derive from the SAME lists the 'caught' section renders
    # (len(data['stuffers']) / len(data['honeypots']) were set in compute_ranking),
    # so the numbers and the cards can never disagree.
    stats = f"""
      <div class="stats">
        <div class="stat"><span class="num">{data['n_pool']}</span><span class="lbl">candidates ranked</span></div>
        <div class="stat"><span class="num">{secs}s</span><span class="lbl">real pipeline · CPU-only</span></div>
        <div class="stat amber"><span class="num">{data['n_stuffers']}</span><span class="lbl">keyword-stuffers caught</span></div>
        <div class="stat red"><span class="num">{data['n_honeypots']}</span><span class="lbl">honeypots floored</span></div>
      </div>"""

    cards = []
    for c in data["top"]:
        cards.append(f"""
        <div class="card">
          <div class="rank">#{c['rank']}</div>
          <div class="body">
            <div class="who"><h3>{_esc(c['name'])}</h3>
              <p>{_esc(c['title'])} · {_esc(c['company'])} · {_fmt_years(c['years'])}</p></div>
            <div class="barrow">
              <div class="bar"><span style="width:{c['pct']:.0f}%"></span></div>
              <div class="score">{c['score']:.3f}</div>
            </div>
            <p class="reason">{_esc(c['reasoning'])}</p>
          </div>
        </div>""")

    caught = []
    for h in data["honeypots"]:
        caught.append(f"""
        <div class="card floored">
          <span class="badge floor">FLOORED · HONEYPOT</span>
          <h3>{_esc(h['name'])}</h3>
          <p class="sub">{_esc(h['title'])} · {_esc(h['company'])}</p>
          <p class="reason">{_esc(h['reason'])}</p>
        </div>""")
    for s in data["stuffers"]:
        caught.append(f"""
        <div class="card flagged">
          <span class="badge flag">FLAGGED · KEYWORD-STUFFER</span>
          <h3>{_esc(s['name'])}</h3>
          <p class="sub">{_esc(s['title'])} · {_esc(s['company'])}</p>
          <p class="reason">{_esc(s['reason'])}</p>
        </div>""")

    return f"""
      {stats}
      {render_jd()}
      <div class="sec">
        <h2>Top {len(data['top'])} by fit</h2>
        <span class="hint">ranked on career substance — every line is grounded in the candidate's own profile</span>
      </div>
      <div class="cards">{''.join(cards)}</div>
      <div class="sec caught-title">
        <h2>⚠ Caliber caught these</h2>
        <span class="hint">impostors the pipeline rejected — not by keyword, by internal evidence</span>
      </div>
      <div class="caught">{''.join(caught)}</div>
      <div class="foot">
        Real <b>src/caliber</b> pipeline · semantic retrieval + BM25 + gated structured
        features + cross-encoder rerank + behavioural multiplier + honeypot floor ·
        <b>zero network</b>, <b>CPU-only</b>, fully deterministic.
      </div>"""


def _hero_block() -> str:
    """The branded hero markup (no .cal wrapper — the caller wraps)."""
    return """
      <div class="hero">
        <div class="glow"></div>
        <span class="kicker">Redrob · India Runs · Track 1</span>
        <h1>Caliber</h1>
        <p class="tag">Ranking candidates by <em>fit</em>, not keywords.</p>
      </div>"""


def landing_html() -> str:
    """Pre-click view: hero + the JD panel + a call-to-action hint. NO results yet —
    those are gated behind the button click (which renders the pre-warmed cache
    instantly)."""
    return _css() + f"""
    <div class="cal">
      {_hero_block()}
      {render_jd()}
      <p class="cta-hint">One pool of 150 candidates. Click <b>Rank</b> to watch Caliber
      separate the genuine fits from the keyword-stuffers and honeypots — instantly.</p>
    </div>"""


def results_html() -> str:
    """Button handler — render the (pre-warmed, cached) results. Returns the hero +
    stats strip + JD + ranked cards + 'caught these'."""
    data = compute_ranking()
    return _css() + f"""
    <div class="cal">
      {_hero_block()}
      {render_results_body(data)}
    </div>"""


# --------------------------------------------------------------------------- #
# Pre-warm: compute the ranking ONCE at startup so the first click is instant.
# Guarded — if the model download / scoring fails here, the button still works
# (it will compute lazily and surface any error to the user).
# --------------------------------------------------------------------------- #
try:
    compute_ranking()
    print("[app] pipeline pre-warmed — results cached, clicks are instant.")
except Exception as exc:  # pragma: no cover - defensive startup guard
    print(f"[app] pre-warm skipped ({type(exc).__name__}: {exc}); will compute on first click.")


# --------------------------------------------------------------------------- #
# Gradio app.
# --------------------------------------------------------------------------- #
_PAGE_CSS = f"""
.gradio-container {{ background: {TOKENS['bg']} !important; max-width: 100% !important; }}
.gradio-container .prose {{ color: {TOKENS['text']}; }}
footer {{ display: none !important; }}
/* Center the call-to-action button in the normal page flow (the gr.Row is a flex
   container; without this its scale=0 child pins to the left edge). */
#rankrow {{ justify-content: center !important; margin: 8px auto 40px !important; }}
#rankbtn {{
  background: linear-gradient(90deg, {TOKENS['accent_a']}, {TOKENS['accent_b']}) !important;
  border: none !important; color: #fff !important; font-weight: 700 !important;
  font-size: 16px !important; border-radius: 999px !important; padding: 14px 30px !important;
  box-shadow: 0 8px 30px rgba(124,58,237,.35) !important;
}}
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Caliber — fit, not keywords", css=_PAGE_CSS,
                   theme=gr.themes.Base()) as demo:
        # Landing: hero + JD panel + the button. Results stay hidden until the click.
        out = gr.HTML(landing_html())
        with gr.Row(elem_id="rankrow"):
            btn = gr.Button("⚡  Rank the 150 candidates", elem_id="rankbtn", scale=0)
        # Click renders the pre-warmed, cached results — returns instantly.
        btn.click(fn=results_html, inputs=None, outputs=out)
    return demo


demo = build_demo()


if __name__ == "__main__":
    # server_name="0.0.0.0" binds all interfaces so the app is reachable from a
    # Windows browser at http://localhost:7860 when it runs inside WSL (Windows
    # forwards localhost to the WSL VM only for 0.0.0.0-bound listeners; a
    # 127.0.0.1 bind is invisible to it). This is also the correct bind for HF
    # Spaces. We deliberately do NOT use share=True — binding 0.0.0.0 is the local
    # fix and needs no Gradio relay / network.
    demo.launch(server_name="0.0.0.0", server_port=7860)
