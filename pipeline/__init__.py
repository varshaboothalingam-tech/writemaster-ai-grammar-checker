"""pipeline — AI-first grammar correction pipeline (feeders + Gemini brain).

    rule_detector  → dependency-light candidate generator (stdlib + data/ only)
    aggregator     → merge candidates, align spans, dedup, confidence
    ai_analyzer    → Gemini full-context analysis → strict JSON
    ai_verifier    → Gemini second-pass verification (accept/reject/uncertain)
    ai_core        → orchestrator: text → candidates → analyze → verify → final
    feedback       → user feedback store + quality gate
"""

__all__ = [
    "rule_detector",
    "aggregator",
    "ai_analyzer",
    "ai_verifier",
    "ai_core",
    "feedback",
]
