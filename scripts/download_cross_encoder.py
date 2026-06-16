"""OFFLINE one-time fetch of the cross-encoder reranker into ``models/``.

Network is allowed during precompute (this script) but FORBIDDEN at rank time.
We therefore download ``cross-encoder/ms-marco-MiniLM-L-6-v2`` once here and save
the full model (weights + tokenizer + config) into
``config.CROSS_ENCODER_MODEL_DIR``. At rank time, ``cross_encoder.load_cross_encoder``
loads only from that local directory — the saved bytes are the determinism pin,
so no revision lookup or network call ever happens online.

Run once (precompute.py should invoke this, or run it directly):

    python scripts/download_cross_encoder.py

Idempotent: if the local dir already looks populated, it does nothing unless
``--force`` is passed.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make ``src/`` importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from caliber import config  # noqa: E402


def _looks_populated(d: Path) -> bool:
    return d.is_dir() and (d / "config.json").exists()


def main(force: bool = False) -> None:
    dest = config.CROSS_ENCODER_MODEL_DIR
    if _looks_populated(dest) and not force:
        print(f"[download_cross_encoder] already present at {dest} — skipping "
              f"(use --force to re-download).")
        return

    # Imported lazily so the rest of the package never needs the heavy deps just
    # to read config.
    from sentence_transformers import CrossEncoder

    print(f"[download_cross_encoder] fetching {config.CROSS_ENCODER_MODEL_NAME} ...")
    model = CrossEncoder(config.CROSS_ENCODER_MODEL_NAME, device="cpu")

    dest.mkdir(parents=True, exist_ok=True)
    # CrossEncoder.save writes the underlying HF model + tokenizer + config.
    model.save(str(dest))
    print(f"[download_cross_encoder] saved to {dest}")


if __name__ == "__main__":
    main(force="--force" in sys.argv[1:])
