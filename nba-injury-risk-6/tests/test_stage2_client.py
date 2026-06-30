"""Tests for Stage 2 infrastructure that doesn't need the network:
retry/backoff behavior and the Gate-2 audit over a synthetic cache.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import nba_injury.nba_client as client  # noqa: E402


def test_with_retry_succeeds_first_try(monkeypatch):
    monkeypatch.setattr(client.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return "ok"

    assert client.with_retry(fn, what="t") == "ok"
    assert calls["n"] == 1


def test_with_retry_recovers_after_failures(monkeypatch):
    monkeypatch.setattr(client.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("transient")
        return "recovered"

    assert client.with_retry(fn, what="t") == "recovered"
    assert calls["n"] == 3


def test_with_retry_raises_after_exhaustion(monkeypatch):
    monkeypatch.setattr(client.time, "sleep", lambda *_: None)

    def fn():
        raise ConnectionError("always down")

    try:
        client.with_retry(fn, what="t")
        assert False, "should have raised"
    except RuntimeError as exc:
        assert "exhausted" in str(exc)


def test_seasons_span_is_ten():
    assert len(client.SEASONS) == 10
    assert client.SEASONS[0] == "2015-16"
    assert client.SEASONS[-1] == "2024-25"
