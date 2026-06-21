"""Network-isolation contract for the ONLINE/judged model loaders.

A single hit to the HuggingFace hub from the judged ``rank.py`` sandbox is a
Stage-3 DISQUALIFICATION (CLAUDE.md hard constraints). Both online-path loaders —
``embeddings.load_model`` and ``cross_encoder.load_cross_encoder`` — therefore
promise the same fail-closed contract:

    when ``CALIBER_ALLOW_MODEL_DOWNLOAD`` is unset (the judged condition) they
    (a) hard-lock HF/transformers to offline BEFORE importing them, and
    (b) load ONLY from the local cache dir, raising a pointed error if it is
        missing rather than reaching for the network.

These tests lock that in for BOTH loaders, three ways:
  1. fail-closed on a missing model dir (raises, never constructs);
  2. the offline env lock (HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE) is set;
  3. no network call — the model constructor is only ever invoked with the LOCAL
     path, never the remote repo id that would trigger a hub fetch.

They are fully offline and deterministic: the real models are never loaded and
the network is never touched. The seam is the ``SentenceTransformer`` /
``CrossEncoder`` constructor (the ONLY place either loader can initiate a fetch —
sentence-transformers only calls ``snapshot_download`` when its first arg is NOT
a local dir), monkeypatched to a spy that records its path arg and explodes if
handed a remote repo id. ``CALIBER_ALLOW_MODEL_DOWNLOAD`` is the precompute-only
download opt-in; we assert the NOT-allow-download (judged) branch throughout.
"""

import os
import types

import pytest

from caliber import config, cross_encoder, embeddings


@pytest.fixture
def online_env(monkeypatch):
    """Put the process in the JUDGED online condition and guarantee no env leak.

    - ``CALIBER_ALLOW_MODEL_DOWNLOAD`` unset  -> the not-allow-download branch
      (rank.py), NOT the precompute download branch.
    - ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE`` unset at start, so a test can
      prove the loader is the thing that SETS them.

    The loaders mutate ``os.environ`` directly (not via monkeypatch), so without
    this fixture they would leak ``HF_HUB_OFFLINE=1`` etc. into the rest of the
    suite. ``monkeypatch.delenv`` records the pre-test state and restores it at
    teardown regardless of what the loader did -> no leakage.
    """
    monkeypatch.delenv("CALIBER_ALLOW_MODEL_DOWNLOAD", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    return monkeypatch


def _forbid_remote_spy(remote_repo_id, calls, make_instance):
    """Build a constructor stand-in that records its path and bans remote ids.

    Returns a callable usable as ``SentenceTransformer`` / ``CrossEncoder``: it
    appends the first positional arg (the path the loader chose) to ``calls`` and
    raises if that arg is the remote repo id — which, for sentence-transformers,
    is precisely what triggers a ``snapshot_download`` (a network call). A local
    directory path never fetches, so it returns a lightweight fake instead.
    """
    def spy(name_or_path, *args, **kwargs):
        chosen = str(name_or_path)
        calls.append(chosen)
        if chosen == remote_repo_id:
            raise AssertionError(
                f"online path constructed the model from remote repo id "
                f"{chosen!r} -> this would hit the HuggingFace hub (network)."
            )
        return make_instance()
    return spy


# ===========================================================================
# 1. FAIL-CLOSED ON MISSING MODEL  (raises; never attempts a download)
# ===========================================================================

def test_embeddings_fail_closed_when_model_dir_missing(online_env, tmp_path):
    import sentence_transformers

    online_env.setattr(embeddings, "_MODEL", None)               # bypass singleton
    online_env.setattr(config, "EMBED_MODEL_LOCAL_DIR", tmp_path / "no-such-model")

    # If the loader ever reaches construction here, the test must fail loudly:
    # a missing local dir must raise, NOT fall through to a remote fetch.
    constructed = []
    online_env.setattr(
        sentence_transformers, "SentenceTransformer",
        lambda *a, **k: constructed.append(a) or types.SimpleNamespace(),
    )

    with pytest.raises(RuntimeError) as exc:
        embeddings.load_model()

    msg = str(exc.value)
    # Error must point a reproducer at the offline precompute + the local cache.
    assert "precompute" in msg.lower()
    assert str(config.EMBED_MODEL_LOCAL_DIR) in msg
    assert constructed == []          # never tried to construct/download anything


def test_cross_encoder_fail_closed_when_model_dir_missing(online_env, tmp_path):
    import sentence_transformers

    online_env.setattr(cross_encoder, "_MODEL", None)
    online_env.setattr(config, "CROSS_ENCODER_MODEL_DIR", tmp_path / "no-such-model")

    constructed = []
    online_env.setattr(
        sentence_transformers, "CrossEncoder",
        lambda *a, **k: constructed.append(a) or types.SimpleNamespace(),
    )

    with pytest.raises(FileNotFoundError) as exc:
        cross_encoder.load_cross_encoder()

    msg = str(exc.value)
    # Error must point a reproducer at the offline download script + cache dir.
    assert "download_cross_encoder" in msg
    assert str(config.CROSS_ENCODER_MODEL_DIR) in msg
    assert constructed == []


def test_cross_encoder_fail_closed_when_config_json_absent(online_env, tmp_path):
    """A present-but-incomplete dir (partial/corrupt download) still fails closed.

    The cross-encoder guard checks for ``config.json`` specifically, so a dir that
    exists but is missing weights/tokenizer must NOT be treated as cached.
    """
    import sentence_transformers

    empty_dir = tmp_path / "ce-half-downloaded"
    empty_dir.mkdir()                                   # exists, but no config.json

    online_env.setattr(cross_encoder, "_MODEL", None)
    online_env.setattr(config, "CROSS_ENCODER_MODEL_DIR", empty_dir)

    constructed = []
    online_env.setattr(
        sentence_transformers, "CrossEncoder",
        lambda *a, **k: constructed.append(a) or types.SimpleNamespace(),
    )

    with pytest.raises(FileNotFoundError):
        cross_encoder.load_cross_encoder()
    assert constructed == []


# ===========================================================================
# 2. OFFLINE ENV LOCK IS SET  (defense-in-depth: HF/transformers forced offline)
# ===========================================================================

def test_embeddings_sets_offline_env_lock(online_env, tmp_path):
    online_env.setattr(embeddings, "_MODEL", None)
    online_env.setattr(config, "EMBED_MODEL_LOCAL_DIR", tmp_path / "missing")

    # The guard sets the offline lock BEFORE the existence check, so it is set
    # even on the raising path. Start state is unset (online_env), proving the
    # loader is what sets it.
    assert "HF_HUB_OFFLINE" not in os.environ
    with pytest.raises(RuntimeError):
        embeddings.load_model()
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def test_cross_encoder_sets_offline_env_lock(online_env, tmp_path):
    online_env.setattr(cross_encoder, "_MODEL", None)
    online_env.setattr(config, "CROSS_ENCODER_MODEL_DIR", tmp_path / "missing")

    assert "TRANSFORMERS_OFFLINE" not in os.environ
    with pytest.raises(FileNotFoundError):
        cross_encoder.load_cross_encoder()
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"


# ===========================================================================
# 3. NO NETWORK CALL  (constructor only ever sees the LOCAL path)
# ===========================================================================

def test_embeddings_constructs_only_from_local_path(online_env, tmp_path):
    """With a cache present, the online loader builds from the LOCAL dir only.

    A local-dir arg to ``SentenceTransformer`` never triggers ``snapshot_download``;
    the remote repo id (``EMBED_MODEL_NAME``) would. The spy bans the latter, so a
    clean return proves the online path stayed entirely on disk.
    """
    import sentence_transformers

    local_dir = tmp_path / "bge-small-en-v1.5"
    local_dir.mkdir()                                   # make the cache "present"
    online_env.setattr(config, "EMBED_MODEL_LOCAL_DIR", local_dir)
    online_env.setattr(embeddings, "_MODEL", None)

    calls = []
    online_env.setattr(
        sentence_transformers, "SentenceTransformer",
        _forbid_remote_spy(config.EMBED_MODEL_NAME, calls,
                           lambda: types.SimpleNamespace()),
    )

    model = embeddings.load_model()

    assert model is not None
    assert calls == [str(local_dir)]                    # exactly one ctor, LOCAL path
    assert config.EMBED_MODEL_NAME not in calls         # never the remote repo id
    assert model.max_seq_length == config.EMBED_MAX_SEQ_LENGTH  # loader finished


def test_cross_encoder_constructs_only_from_local_path(online_env, tmp_path):
    """Same proof for the cross-encoder loader."""
    import sentence_transformers

    model_dir = tmp_path / "ms-marco-MiniLM-L-6-v2"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")   # cache "present"
    online_env.setattr(config, "CROSS_ENCODER_MODEL_DIR", model_dir)
    online_env.setattr(cross_encoder, "_MODEL", None)

    def make_ce():
        # load_cross_encoder() calls model.model.eval() after construction.
        return types.SimpleNamespace(model=types.SimpleNamespace(eval=lambda: None))

    calls = []
    online_env.setattr(
        sentence_transformers, "CrossEncoder",
        _forbid_remote_spy(config.CROSS_ENCODER_MODEL_NAME, calls, make_ce),
    )

    model = cross_encoder.load_cross_encoder()

    assert model is not None
    assert calls == [str(model_dir)]                    # exactly one ctor, LOCAL path
    assert config.CROSS_ENCODER_MODEL_NAME not in calls  # never the remote repo id
