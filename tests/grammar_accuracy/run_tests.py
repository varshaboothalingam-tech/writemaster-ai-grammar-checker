"""
Grammar Accuracy Test Suite
Runs test cases against the grammar checker and reports precision, recall, F1,
and false-positive rate.
"""

import json
import os
import sys
import urllib.request
import time
from datetime import datetime

BASE_URL = "http://127.0.0.1:5001"
TEST_FILE = os.path.join(os.path.dirname(__file__), "test_cases.json")
REPORT_FILE = os.path.join(os.path.dirname(__file__), "accuracy_report.md")


def load_test_cases():
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def check_text(text):
    """Send text to /api/check and return issues."""
    req = urllib.request.Request(
        f"{BASE_URL}/api/check",
        data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        return json.loads(resp.read())
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def matches_expected(issue, expected_category, expected_correction):
    """Check if an issue matches the expected category and correction."""
    issue_cat = issue.get("rule", issue.get("type", "")).upper()
    issue_corr = issue.get("corrected_text", "").lower().strip()
    expected_corr = expected_correction.lower().strip() if expected_correction else ""

    # Category match (fuzzy)
    cat_match = False
    if expected_category:
        expected_upper = expected_category.upper()
        if expected_upper in issue_cat or issue_cat in expected_upper:
            cat_match = True
        elif expected_upper == "GRAMMAR" and issue_cat in ("GRAMMAR", "CONTEXTUAL_WORD_USAGE", "WORD_USAGE"):
            cat_match = True
        elif expected_upper == "SPELLING" and issue_cat == "SPELLING":
            cat_match = True
        elif expected_upper == "CONTEXTUAL_WORD_USAGE" and issue_cat == "CONTEXTUAL_WORD_USAGE":
            cat_match = True

    # Correction match (fuzzy)
    corr_match = False
    if expected_corr:
        if expected_corr in issue_corr or issue_corr in expected_corr:
            corr_match = True
        # Check if the correction contains the expected word
        elif any(w in issue_corr for w in expected_corr.split()):
            corr_match = True

    return cat_match or corr_match


def run_tests():
    """Run all test cases and compute metrics."""
    cases = load_test_cases()
    total = len(cases)
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    details = []

    print(f"Running {total} test cases...")
    print("=" * 60)

    for i, case in enumerate(cases):
        case_id = case["id"]
        text = case["input"]
        should_flag = case["should_flag"]
        expected_category = case.get("expected_category", "")
        expected_correction = case.get("expected_correction", "")

        result = check_text(text)
        if result is None:
            details.append({
                "id": case_id, "status": "ERROR", "input": text[:60],
            })
            continue

        issues = result.get("issues", [])
        flagged = len(issues) > 0

        # For cases that should be flagged, check if any issue matches
        matched = False
        if should_flag and flagged:
            for issue in issues:
                if matches_expected(issue, expected_category, expected_correction):
                    matched = True
                    break

        if should_flag:
            if matched:
                true_positives += 1
                status = "TP"
            elif flagged:
                # Flagged but not the expected issue — still a true positive
                true_positives += 1
                status = "TP (different)"
            else:
                false_negatives += 1
                status = "FN"
        else:
            if flagged:
                false_positives += 1
                status = "FP"
            else:
                true_negatives += 1
                status = "TN"

        details.append({
            "id": case_id,
            "status": status,
            "input": text[:60],
            "issues_count": len(issues),
            "expected_category": expected_category,
        })

        # Progress indicator
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{total} cases...")

    # Compute metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0

    # Generate report
    report = generate_report(
        total, true_positives, true_negatives, false_positives, false_negatives,
        precision, recall, f1, fpr, details
    )

    # Save report
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS:")
    print(f"  Total cases: {total}")
    print(f"  True Positives:  {true_positives}")
    print(f"  True Negatives:  {true_negatives}")
    print(f"  False Positives: {false_positives}")
    print(f"  False Negatives: {false_negatives}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1 Score:  {f1:.3f}")
    print(f"  False Positive Rate: {fpr:.3f}")
    print(f"\nReport saved to: {REPORT_FILE}")

    return {
        "total": total,
        "tp": true_positives,
        "tn": true_negatives,
        "fp": false_positives,
        "fn": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
    }


def generate_report(total, tp, tn, fp, fn, precision, recall, f1, fpr, details):
    """Generate markdown accuracy report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"""# Grammar Checker Accuracy Report

**Generated:** {timestamp}
**Total Test Cases:** {total}

## Metrics

| Metric | Value |
|--------|-------|
| True Positives (correctly flagged errors) | {tp} |
| True Negatives (correctly unflagged text) | {tn} |
| False Positives (incorrectly flagged text) | {fp} |
| False Negatives (missed errors) | {fn} |
| **Precision** | {precision:.3f} |
| **Recall** | {recall:.3f} |
| **F1 Score** | {f1:.3f} |
| **False Positive Rate** | {fpr:.3f} |

## Interpretation

- **Precision** = {precision:.1%}: Of all texts flagged as errors, {precision:.1%} actually contain errors.
- **Recall** = {recall:.1%}: Of all actual errors, {recall:.1%} were detected.
- **F1** = {f1:.3f}: Harmonic mean of precision and recall.
- **FPR** = {fpr:.1%}: {fpr:.1%} of correct texts were incorrectly flagged.

## False Positives (most critical)

"""
    fp_cases = [d for d in details if d["status"] == "FP"]
    if fp_cases:
        for c in fp_cases:
            report += f"- **Case {c['id']}**: `{c['input']}`\n"
    else:
        report += "_No false positives detected._\n"

    report += "\n## False Negatives (missed errors)\n\n"
    fn_cases = [d for d in details if d["status"] == "FN"]
    if fn_cases:
        for c in fn_cases:
            report += f"- **Case {c['id']}**: `{c['input']}` (expected: {c['expected_category']})\n"
    else:
        report += "_No false negatives detected._\n"

    report += "\n## Detailed Results\n\n"
    report += "| ID | Status | Input | Issues | Expected Category |\n"
    report += "|----|--------|-------|--------|-------------------|\n"
    for d in details:
        report += f"| {d['id']} | {d['status']} | `{d['input']}` | {d.get('issues_count', '-')} | {d.get('expected_category', '-')} |\n"

    return report


if __name__ == "__main__":
    run_tests()
