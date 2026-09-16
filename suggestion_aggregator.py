"""
Suggestion Aggregator — single canonical output from multiple engines.

Collects raw candidates from WritingAnalyzer, GrammarChecker, and AIEngine,
normalizes to a shared schema, runs through the suggestion validator, and
returns one ranked list for the frontend.

  WritingAnalyzer ─┐
  GrammarChecker  ─┼──→ SuggestionAggregator → dedupe → conflict resolution
  AIEngine        ─┘                        → confidence filter → ranked final list
"""

import re
import os
from typing import List, Dict, Optional


# Canonical candidate schema (must match suggestion_validator expectations)
CANONICAL_FIELDS = (
    "rule_id", "category", "original_text", "replacement",
    "explanation", "confidence", "severity",
    "start_position", "end_position",
)


class SuggestionAggregator:
    """Collects, normalizes, validates, and deduplicates suggestions
    from all engines into one canonical list."""

    def __init__(self, writing_analyzer, grammar_checker=None, ai_engine=None,
                 data_dir: str = "data"):
        self.wa = writing_analyzer
        self.gc = grammar_checker
        self.ai = ai_engine
        self._data_dir = data_dir
        self._user_dict_path = os.path.join(data_dir, "user_dictionary.json")

        try:
            from suggestion_validator import validate_candidates
            self._validate = validate_candidates
        except ImportError:
            self._validate = None

    def aggregate(self, text: str) -> Dict:
        """Run all engines, normalize, validate, and return canonical output.

        Returns:
            {
                "issues": [...],       # normalized issue list for frontend
                "readability": {...},
                "advanced": {...},
                "stats": {...},
            }
        """
        if not text or not text.strip():
            return {
                "issues": [],
                "readability": self.wa.readability.calculate(""),
                "advanced": {"advanced_issues": [], "word_count": 0},
                "stats": {"word_count": 0, "sentence_count": 0, "error_count": 0},
            }

        # 1. Collect raw candidates from all engines
        all_candidates: List[Dict] = []

        # WritingAnalyzer (primary engine — runs spelling, grammar, contextual, etc.)
        wa_result = self.wa.analyze(text, config={"min_confidence": 0.50})
        all_candidates.extend(wa_result.get("errors", []))
        readability = wa_result.get("readability", {})
        stats = wa_result.get("stats", {})

        # GrammarChecker (legacy engine — used for /api/correct, optional for /api/check)
        if self.gc:
            try:
                gc_errors = self.gc.check(text)
                all_candidates.extend(self._normalize_grammar_checker(gc_errors, text))
            except Exception:
                pass

        # AIEngine advanced analysis (cliches, redundancy, passive voice)
        advanced = {"advanced_issues": [], "word_count": 0}
        if self.ai:
            try:
                adv = self.ai.advanced_analysis(text)
                advanced = adv
                all_candidates.extend(self._normalize_ai_engine(adv, text))
            except Exception:
                pass

        # 2. Run through suggestion validator
        if self._validate:
            all_candidates = self._validate(
                all_candidates,
                text,
                supporting_signals=all_candidates,
                user_dict_path=self._user_dict_path,
            )

        # 3. Sort by severity and confidence
        severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        all_candidates.sort(
            key=lambda e: (
                -severity_order.get(e.get("severity", "MEDIUM"), 1),
                -e.get("confidence", 0),
            )
        )

        return {
            "issues": all_candidates,
            "readability": readability,
            "advanced": advanced,
            "stats": stats,
        }

    def _normalize_grammar_checker(self, gc_errors: List[Dict], text: str) -> List[Dict]:
        """Normalize GrammarChecker output to canonical schema."""
        normalized = []
        for err in gc_errors:
            # GrammarChecker uses different field names
            original = err.get("incorrect", err.get("original", ""))
            replacement = err.get("correct", err.get("suggestion", ""))
            if not original or not replacement:
                continue

            # Find position in text
            start = err.get("position", 0)
            if start == 0 and original:
                idx = text.lower().find(original.lower())
                if idx >= 0:
                    start = idx
            end = start + len(original)

            normalized.append({
                "rule_id": "gc_" + err.get("category", "grammar").lower().replace(" ", "_"),
                "category": _map_gc_category(err.get("category", "GRAMMAR")),
                "original_text": original,
                "replacement": replacement,
                "explanation": err.get("message", err.get("explanation", "")),
                "confidence": float(err.get("confidence", 0.70)),
                "severity": _map_gc_severity(err.get("severity", "medium")),
                "start_position": start,
                "end_position": end,
                "error_type": err.get("type", "grammar"),
                "message": err.get("message", ""),
                "context": err.get("context", text[max(0, start - 50):end + 50]),
            })
        return normalized

    def _normalize_ai_engine(self, advanced: Dict, text: str) -> List[Dict]:
        """Normalize AIEngine advanced_analysis output to canonical schema."""
        normalized = []
        issues = advanced.get("advanced_issues", [])
        for issue in issues:
            original = issue.get("original", "")
            suggestions = issue.get("suggestions", [])
            replacement = suggestions[0] if suggestions else ""

            # Find position in text
            idx = text.lower().find(original.lower()) if original else -1
            start = idx if idx >= 0 else 0
            end = start + len(original)

            # Map AI engine types to categories
            category_map = {
                "cliche": "STYLE",
                "redundancy": "REDUNDANCY",
                "passive_voice": "STYLE",
                "filler_words": "STYLE",
                "weak_words": "STYLE",
            }
            category = category_map.get(issue.get("type", ""), "STYLE")

            normalized.append({
                "rule_id": "ai_" + issue.get("type", "style").lower().replace(" ", "_"),
                "category": category,
                "original_text": original,
                "replacement": replacement,
                "explanation": issue.get("message", ""),
                "confidence": 0.65,  # AI engine suggestions are style-level
                "severity": "LOW",
                "start_position": start,
                "end_position": end,
                "error_type": category.lower(),
                "message": issue.get("message", ""),
                "context": text[max(0, start - 50):end + 50],
            })
        return normalized


def _map_gc_category(cat: str) -> str:
    """Map GrammarChecker categories to canonical categories."""
    cat_lower = cat.lower() if cat else ""
    if "spelling" in cat_lower:
        return "SPELLING"
    if "grammar" in cat_lower:
        return "GRAMMAR"
    if "punctuation" in cat_lower:
        return "PUNCTUATION"
    if "capital" in cat_lower:
        return "CAPITALIZATION"
    if "context" in cat_lower or "usage" in cat_lower:
        return "WORD_USAGE"
    return "GRAMMAR"


def _map_gc_severity(sev: str) -> str:
    """Map GrammarChecker severity to canonical severity."""
    if not sev:
        return "MEDIUM"
    sev_lower = sev.lower()
    if sev_lower in ("high", "error"):
        return "HIGH"
    if sev_lower in ("medium", "warning"):
        return "MEDIUM"
    if sev_lower in ("low", "info", "suggestion"):
        return "LOW"
    return "MEDIUM"
