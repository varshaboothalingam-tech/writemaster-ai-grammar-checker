"""
Benchmark runner: tests the current grammar checker against 1,000 sentences.
Reports precision, recall, F1, FPR by category.
"""
import json
import requests
import time
import sys
from collections import defaultdict

API_URL = "http://127.0.0.1:5001/api/check-v4"
BENCHMARK_PATH = r"C:\Users\hemal\OneDrive\Documents\grammar_checker\tests\grammar_accuracy\benchmark_1000.json"


def run_benchmark():
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print(f"Running benchmark: {len(cases)} sentences")
    print("=" * 60)

    results = []
    errors_by_category = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})

    for i, case in enumerate(cases):
        text = case["text"]
        should_flag = case.get("should_flag", False)
        expected_errors = case.get("expected_errors", [])
        categories = case.get("categories", [])

        try:
            resp = requests.post(API_URL, json={"text": text}, timeout=30)
            data = resp.json()
            issues = data.get("issues", [])
        except Exception as e:
            print(f"  ERROR on case {case.get('id', i)}: {e}")
            issues = []

        # Determine if system flagged this sentence
        system_flagged = len(issues) > 0

        # Determine if it's a true positive, false positive, etc.
        if should_flag and system_flagged:
            result = "TP"
        elif should_flag and not system_flagged:
            result = "FN"
        elif not should_flag and system_flagged:
            result = "FP"
        else:
            result = "TN"

        results.append({
            "id": case.get("id", i),
            "text": text[:80],
            "should_flag": should_flag,
            "system_flagged": system_flagged,
            "result": result,
            "num_issues": len(issues),
            "categories": categories,
            "issues": [{"word": iss.get("word", ""), "rule": iss.get("rule", ""), "message": iss.get("message", "")[:60]} for iss in issues[:3]],
        })

        # Update per-category stats
        for cat in categories:
            if should_flag and system_flagged:
                errors_by_category[cat]["tp"] += 1
            elif should_flag and not system_flagged:
                errors_by_category[cat]["fn"] += 1
            elif not should_flag and system_flagged:
                errors_by_category[cat]["fp"] += 1
            else:
                errors_by_category[cat]["tn"] += 1

        # Progress
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(cases)}...")

    # Overall stats
    tp = sum(1 for r in results if r["result"] == "TP")
    fp = sum(1 for r in results if r["result"] == "FP")
    fn = sum(1 for r in results if r["result"] == "FN")
    tn = sum(1 for r in results if r["result"] == "TN")

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 0.001)
    fpr = fp / max(fp + tn, 1)

    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    print(f"  True Positives:  {tp}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Negatives:  {tn}")
    print(f"  Precision:       {precision:.3f} ({precision*100:.1f}%)")
    print(f"  Recall:          {recall:.3f} ({recall*100:.1f}%)")
    print(f"  F1 Score:        {f1:.3f}")
    print(f"  False Pos Rate:  {fpr:.3f} ({fpr*100:.1f}%)")

    # Per-category results
    print("\n" + "=" * 60)
    print("PER-CATEGORY RESULTS")
    print("=" * 60)
    for cat in sorted(errors_by_category.keys()):
        stats = errors_by_category[cat]
        cat_tp = stats["tp"]
        cat_fp = stats["fp"]
        cat_fn = stats["fn"]
        cat_tn = stats["tn"]
        cat_prec = cat_tp / max(cat_tp + cat_fp, 1)
        cat_recall = cat_tp / max(cat_tp + cat_fn, 1)
        cat_f1 = 2 * cat_prec * cat_recall / max(cat_prec + cat_recall, 0.001)
        print(f"  {cat:25s} P={cat_prec:.2f} R={cat_recall:.2f} F1={cat_f1:.2f} FP={cat_fp} FN={cat_fn}")

    # Show false positives
    fps = [r for r in results if r["result"] == "FP"]
    if fps:
        print(f"\n{'=' * 60}")
        print(f"FALSE POSITIVES ({len(fps)})")
        print("=" * 60)
        for r in fps[:30]:
            print(f"  [{r['id']}] {r['text']}")
            for iss in r["issues"]:
                print(f"       -> {iss['rule']}: {iss['message']}")
            print()

    # Save detailed results
    report = {
        "total": len(cases),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "by_category": {cat: dict(stats) for cat, stats in errors_by_category.items()},
        "details": results,
    }
    out_path = r"C:\Users\hemal\OneDrive\Documents\grammar_checker\tests\grammar_accuracy\benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    run_benchmark()
