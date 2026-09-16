"""API endpoint tests (spec §10): /api/check (approved-only), /api/health,
/api/validate, and the frontend issue 'type' mapping for the §12 stats."""
import os

import pytest

os.environ.setdefault("AI_PROVIDER", "none")

from app import app  # noqa: E402


@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()


def _issues(r):
    return r.get_json()["issues"]


def test_health_reports_v4_and_ai_state(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    d = r.get_json()
    assert d["status"] == "ok"
    assert d["pipeline"] == "v4"
    assert d["v4_pipeline_available"] is True
    assert d["ai"]["enabled"] is False


def test_check_returns_only_approved_errors(client):
    r = client.post("/api/check", json={
        "text": "Yesterday, our sales team has attended an important meeting."})
    assert r.status_code == 200
    issues = _issues(r)
    assert any(i["rule_id"] == "TENSE_CONSISTENCY"
               and i["original"] == "has attended" and i["replacement"] == "attended"
               for i in issues)


def test_check_clean_sentence_no_errors(client):
    r = client.post("/api/check", json={"text": "Some information was incorrect."})
    assert r.status_code == 200
    assert _issues(r) == []


def test_check_v4_alias_has_type_buckets(client):
    r = client.post("/api/check-v4", json={"text": "She recieve the package yesterday."})
    assert r.status_code == 200
    issues = _issues(r)
    assert any(i["rule_id"] == "SPELL_MISSPELLING" and i["type"] == "spelling" for i in issues)
    assert all(i.get("type") in ("spelling", "grammar", "style", "punctuation") for i in issues)


def test_validate_matches_candidate(client):
    r = client.post("/api/validate", json={
        "text": "The meeting were scheduled for Monday.",
        "original": "were",
        "replacement": "was",
    })
    assert r.status_code == 200
    d = r.get_json()
    assert d["validated"] is True
    assert d["matched_candidate"]["rule_id"] == "SVA"


def test_validate_rejected_candidate_not_exposed(client):
    r = client.post("/api/validate", json={
        "text": "He will send the missing files.",
        "original": "will",
        "replacement": "would",
    })
    d = r.get_json()
    assert d["validated"] is True
    assert d["approved_errors"] == []
    assert d["matched_candidate"] is None


def test_ai_decision_log_writes_jsonl_only_when_enabled(tmp_path, monkeypatch):
    import new_pipeline
    monkeypatch.setattr(new_pipeline, "_AI_LOG_PATH",
                        str(tmp_path / "ai_decisions.jsonl"))
    monkeypatch.setattr(new_pipeline, "_AI_LOG_ENABLED", True)
    new_pipeline._log_ai_decision({"rule": "SVA", "final_decision": "APPROVED"})
    new_pipeline._log_ai_decision({"rule": "TENSE", "final_decision": "REJECTED"})
    lines = (tmp_path / "ai_decisions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert lines[0].split('"final_decision"')[-1].startswith(": \"APPROVED\"")
    # disabled -> no file writes
    monkeypatch.setattr(new_pipeline, "_AI_LOG_ENABLED", False)
    new_pipeline._log_ai_decision({"rule": "X"})
    assert len((tmp_path / "ai_decisions.jsonl").read_text(encoding="utf-8").strip().splitlines()) == 2