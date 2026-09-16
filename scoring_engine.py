"""
Multi-dimension scoring engine.
Scores text on 4 axes: Correctness, Clarity, Engagement, Delivery.
Each score is 0-100 (100 = best).
"""

import math
from typing import List, Dict


CATEGORY_WEIGHTS = {
    "grammar":           {"correctness": 8.0, "clarity": 2.0, "engagement": 0.0, "delivery": 0.0},
    "spelling":          {"correctness": 10.0, "clarity": 1.0, "engagement": 0.0, "delivery": 0.0},
    "punctuation":       {"correctness": 5.0, "clarity": 1.5, "engagement": 0.0, "delivery": 0.0},
    "capitalization":    {"correctness": 4.0, "clarity": 0.5, "engagement": 0.0, "delivery": 0.0},
    "style":             {"correctness": 0.0, "clarity": 6.0, "engagement": 4.0, "delivery": 1.0},
    "contextual":        {"correctness": 5.0, "clarity": 4.0, "engagement": 2.0, "delivery": 0.0},
    "redundancy":        {"correctness": 0.0, "clarity": 7.0, "engagement": 2.0, "delivery": 0.0},
    "cliche":            {"correctness": 0.0, "clarity": 2.0, "engagement": 8.0, "delivery": 1.0},
    "passive_voice":     {"correctness": 0.0, "clarity": 3.0, "engagement": 1.0, "delivery": 4.0},
    "filler_words":      {"correctness": 0.0, "clarity": 6.0, "engagement": 1.0, "delivery": 6.0},
    "wordiness":         {"correctness": 0.0, "clarity": 6.0, "engagement": 1.0, "delivery": 6.0},
    "jargon":            {"correctness": 0.0, "clarity": 5.0, "engagement": 1.0, "delivery": 4.0},
    "sentence_structure": {"correctness": 3.0, "clarity": 4.0, "engagement": 0.0, "delivery": 0.0},
    "fragment":          {"correctness": 6.0, "clarity": 3.0, "engagement": 0.0, "delivery": 0.0},
    "parallelism":       {"correctness": 2.0, "clarity": 5.0, "engagement": 1.0, "delivery": 0.0},
}

SEVERITY_MULTIPLIER = {
    "error": 1.0,
    "warning": 0.6,
    "info": 0.3,
    "suggestion": 0.3,
}


def compute_scores(
    errors: List[Dict],
    readability: Dict,
    word_count: int,
    sentence_count: int,
) -> Dict:
    if not word_count:
        word_count = 1
    if not sentence_count:
        sentence_count = 1

    dim_totals = {"correctness": 0.0, "clarity": 0.0, "engagement": 0.0, "delivery": 0.0}

    for err in errors:
        category = (err.get("category") or err.get("error_type") or "grammar").lower().replace(" ", "_")
        severity = (err.get("severity") or "warning").lower()
        conf = err.get("confidence", 0.8)

        weights = CATEGORY_WEIGHTS.get(category, CATEGORY_WEIGHTS["grammar"])
        mult = SEVERITY_MULTIPLIER.get(severity, 0.6)

        penalty = mult * conf
        for dim in dim_totals:
            w = weights.get(dim, 0.0)
            if w > 0:
                dim_totals[dim] += penalty * w

    # Use errors-per-sentence for normalization (like Grammarly)
    # More sentences = more tolerance, but use sqrt to prevent over-dilution
    errors_per_sentence = sum(dim_totals.values()) / max(1, sentence_count)
    # Scale: 0 errors/sentence = 0 penalty, 1 error/sentence = ~25 penalty, 3+ = ~60+
    normalization = max(1.0, math.sqrt(sentence_count))
    for dim in dim_totals:
        dim_totals[dim] /= normalization

    scores = {}
    for dim in dim_totals:
        raw = 100 - dim_totals[dim]
        scores[dim] = max(0, min(100, round(raw)))

    # Readability-based adjustments
    flesch = readability.get("score", 50)
    passive_pct = readability.get("passive_voice_pct", 0)

    if flesch >= 70:
        scores["clarity"] = min(100, scores["clarity"] + 3)
    elif flesch < 40:
        scores["clarity"] = max(0, scores["clarity"] - 5)

    if passive_pct > 30:
        scores["delivery"] = max(0, scores["delivery"] - 5)

    complex_pct = readability.get("complex_word_pct", 0)
    if complex_pct > 40:
        scores["engagement"] = max(0, scores["engagement"] - 3)

    def label(score):
        if score >= 90:
            return "Excellent"
        elif score >= 75:
            return "Good"
        elif score >= 50:
            return "Fair"
        elif score >= 30:
            return "Needs Work"
        else:
            return "Poor"

    overall = round((scores["correctness"] + scores["clarity"] +
                     scores["engagement"] + scores["delivery"]) / 4)

    return {
        "correctness": {"score": scores["correctness"], "label": label(scores["correctness"])},
        "clarity":     {"score": scores["clarity"],     "label": label(scores["clarity"])},
        "engagement":  {"score": scores["engagement"],   "label": label(scores["engagement"])},
        "delivery":    {"score": scores["delivery"],     "label": label(scores["delivery"])},
        "overall":     {"score": overall,                "label": label(overall)},
    }
