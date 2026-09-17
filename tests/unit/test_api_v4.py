"""Production API contract tests (spec §10): /api/check (v2 error object,
consensus, approved errors) and /api/diag. The legacy app.py /api/health,
/api/check-v4 and /api/validate routes were removed in Phase 16; these tests
target the deployed api/index.py surface only."""
import os

import pytest

os.environ.setdefault("AI_PROVIDER", "none")

from api.index import app  # noqa: E402


@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()


def _errors(r):
    return r.get_json()["errors"]


def test_diag_reports_provider_and_ai_state(client):
    r = client.get("/api/diag")
    assert r.status_code == 200
    d = r.get_json()
    assert "ai_configured" in d
    assert d["ai_provider_env"] == "none"
    assert d["is_enabled"] is False


def test_check_returns_only_approved_errors(client):
    r = client.post("/api/check", json={
        "text": "Yesterday, our sales team has attended an important meeting."})
    assert r.status_code == 200
    errors = _errors(r)
    assert any(i["rule_id"] == "TENSE_CONSISTENCY"
               and i["original"] == "has attended" and i["replacement"] == "attended"
               for i in errors)


def test_check_clean_sentence_no_errors(client):
    r = client.post("/api/check", json={"text": "Some information was incorrect."})
    assert r.status_code == 200
    assert _errors(r) == []


def test_check_error_object_has_type_bucket_schema(client):
    r = client.post("/api/check", json={"text": "She recieve the package yesterday."})
    assert r.status_code == 200
    errors = _errors(r)
    assert any(i["rule_id"] == "SPELLING_DICT" and i["type"] == "spelling"
               and i["category"] == "spelling" and i["wrong"] == "recieve"
               and i["correct"] == "receive" for i in errors)
    assert all(i.get("type") in ("spelling", "grammar", "style", "punctuation")
               and i.get("start") is not None and i.get("end") is not None
               for i in errors)


def test_check_sva_candidate_approved(client):
    r = client.post("/api/check", json={
        "text": "The meeting were scheduled for Monday."})
    assert r.status_code == 200
    d = r.get_json()
    assert any(i["rule_id"] == "SVA" and i["original"] == "were"
               and i["replacement"] == "was" for i in d["errors"])
    assert d["corrected_text"] == "The meeting was scheduled for Monday."


def test_check_non_error_not_exposed(client):
    r = client.post("/api/check", json={"text": "He will send the missing files."})
    assert r.status_code == 200
    d = r.get_json()
    assert d["errors"] == []


def test_check_response_surface(client):
    r = client.post("/api/check", json={"text": "Some information was incorrect."})
    assert r.status_code == 200
    d = r.get_json()
    assert d["success"] is True
    assert d["schema"] == "v2_error_object"
    assert d["pipeline"] == "master"
    assert d["verification"] == "offline"
    assert "quality" in d and "consensus" in d


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