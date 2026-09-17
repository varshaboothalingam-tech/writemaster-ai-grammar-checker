"""Vercel serverless entry point.

The dependency-light AI-first pipeline (pipeline/rule_detector + aggregator +
Gemini analyze/verify) contains NO spaCy / NLTK imports, so it runs inside a
Vercel Python function. The heavy deterministic engines (spaCy v4) are not
importable here — Gemini provides the grammar brain instead.

Endpoints:
    POST /api/check   -> {success, original_text, corrected_text, errors, ...}
    GET  /            -> service banner

Set GEMINI_API_KEY as a Vercel environment variable for full AI analysis.
Without a key the function degrades to the conservative offline rule path.
"""
import json
import os
import time

from flask import Flask, Response, request

from pipeline.ai_core import check_ai_text
from pipeline.ai_analyzer import ai_key_configured

app = Flask(__name__)


@app.route("/", methods=["GET"])
def root():
    return {
        "service": "writemaster-ai-grammar-checker",
        "version": "1.0.0",
        "ai_configured": ai_key_configured(),
        "mode": "serverless",
    }


@app.route("/api/check", methods=["POST"])
def api_check():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "") or "").strip()
    if not text:
        return Response(
            json.dumps({"success": False, "message": "empty text",
                        "issues": [], "errors": [], "corrected_text": ""}),
            status=400, mimetype="application/json")
    try:
        result = check_ai_text(text, use_ai=payload.get("use_ai", True)
                               if ai_key_configured() else False)
        return Response(
            json.dumps({
                "success": True,
                "original_text": result["original_text"],
                "corrected_text": result["corrected_text"],
                "errors": result["errors"],
                "grammar_status": result["grammar_status"],
                "processing_time_ms": result["processing_time_ms"],
            }, ensure_ascii=False),
            status=200, mimetype="application/json")
    except Exception as exc:  # pragma: no cover - defensive
        return Response(
            json.dumps({"success": False, "message": str(exc),
                        "issues": [], "errors": [], "corrected_text": text}),
            status=500, mimetype="application/json")


# Vercel's @vercel/python build serves this WSGI app directly.
wsgi_app = app