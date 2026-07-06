"""Shared fixtures for the regression suite.

These tests capture the verification gates used while fixing real production
failures (parse rate-limit storms, LLM schema drift, batch size caps, cluster
noise handling, SCU column mapping, FDR statistics). Keep them green — every
one of them corresponds to a bug that actually shipped once.

Run with:  uv run --with pytest pytest
"""
import json
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def master_schema_text() -> str:
    return (ROOT / "uap_master_schema.json").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def master_schema(master_schema_text) -> dict:
    return json.loads(master_schema_text)


@pytest.fixture()
def no_sleep(monkeypatch):
    """Disable real backoff waits inside uap_analyzer retry loops."""
    import uap_analyzer as U
    monkeypatch.setattr(U.time, "sleep", lambda *a, **k: None)


@pytest.fixture()
def fake_embedder(monkeypatch):
    """Deterministic tiny embedding model — no sentence-transformers load.

    'disc' and 'disk' are near-identical; 'triangle' is orthogonal.
    """
    import uap_analyzer as U

    class FakeModel:
        _VECS = {
            "disc": [1.0, 0.0, 0.0],
            "disk": [0.999, 0.04, 0.0],
            "triangle": [0.0, 1.0, 0.0],
        }

        def encode_document(self, terms):
            return np.array([self._VECS.get(t, [0.0, 0.0, 1.0]) for t in terms])

    monkeypatch.setattr(U, "get_embed_model", lambda: FakeModel())


@pytest.fixture()
def api_client():
    """FastAPI TestClient with a unique session per test (no cross-test state)."""
    from fastapi.testclient import TestClient
    import api.main as M

    client = TestClient(M.app)
    client.headers["X-Session-ID"] = f"test-{uuid.uuid4().hex[:10]}"
    return client


def make_parser(**kw):
    from uap_analyzer import UAPParser
    return UAPParser(api_key="sk-test-dummy", **kw)


class MockOpenAIFiles:
    def __init__(self, store):
        self.store = store

    def create(self, file, purpose):
        name, buf, _mime = file
        self.store.append((name, buf.read()))
        return type("F", (), {"id": f"file_{len(self.store)}"})()


class MockOpenAIBatches:
    def __init__(self, store):
        self.store = store

    def create(self, input_file_id, endpoint, completion_window):
        self.store.append(input_file_id)
        return type("B", (), {"id": f"batch_{len(self.store)}"})()


def mock_batch_client(files_store, batches_store):
    return type("C", (), {
        "files": MockOpenAIFiles(files_store),
        "batches": MockOpenAIBatches(batches_store),
    })()
