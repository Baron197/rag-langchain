"""Tests for the LangChain variant's /upload endpoint.

Uploaded files are saved into DOCS_DIR, the in-memory index is rebuilt, and the
new content becomes searchable -- all on the keyless `fake` path. DOCS_DIR and
TRACE_DIR are pointed at a temp directory (via env) so the real `data/docs`
corpus and the repo are never touched, and the api module is reloaded so its
singleton picks up that config.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _client(tmp_path, monkeypatch) -> TestClient:
    """A TestClient whose pipeline reads/writes a temp docs + trace dir (keyless)."""
    monkeypatch.setenv("DOCS_DIR", str(tmp_path / "docs"))
    monkeypatch.setenv("TRACE_DIR", str(tmp_path / "traces"))
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
    from src.rag_lc import api
    importlib.reload(api)  # rebuild module-level app + singleton with the new env
    return TestClient(api.app)


def test_upload_saves_indexes_and_is_searchable(tmp_path, monkeypatch):
    """A .md upload is stored, indexed, and then retrievable via /query."""
    client = _client(tmp_path, monkeypatch)
    body = b"# Refunds\n\nRefunds are processed within 14 business days via the billing portal."
    r = client.post("/upload", files=[("files", ("refunds.md", body, "text/markdown"))])
    assert r.status_code == 200, r.text
    data = r.json()
    assert "refunds.md" in data["saved"]
    assert data["indexed_chunks"] > 0
    # The uploaded document is now searchable.
    q = client.post("/query", json={"question": "How long do refunds take?"})
    assert q.status_code == 200
    assert "refunds.md" in {c["source"] for c in q.json()["citations"]}


def test_upload_rejects_unsupported_type(tmp_path, monkeypatch):
    """An unsupported extension is rejected with HTTP 400 (nothing indexed)."""
    client = _client(tmp_path, monkeypatch)
    r = client.post("/upload", files=[("files", ("evil.exe", b"MZ", "application/octet-stream"))])
    assert r.status_code == 400
