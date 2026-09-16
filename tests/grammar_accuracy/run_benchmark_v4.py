"""
run_benchmark_v4.py — honest benchmark for the v4 pipeline.

Runs in-process (no HTTP server). Same TP/FP/FN/TN + P/R/F1/FPR accounting as
the original run_benchmark.py, but reports the v4 pipeline directly and prints
every FP and FN so failures can be triaged.

Usage:
    python tests/grammar_accuracy/run_benchmark_v4.py [path-to-cases.json]
Defaults to tests/grammar_accuracy/test_cases.json
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from new_pipeline import check_v4  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CASES = os.path.join(HERE, "test_cases.json")


def run(cases_path: str):
    with open(cases_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print(f"Running v4 benchmark: {len(cases)} cases from {os.path.basename(cases_path)}")
    print("=" * 72)

    results = []
    by_cat = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})

    for i, case in enumerate(cases):
        text = case.get("input") or case.get("text", "")
        should_flag = bool(case.get("should_flag", False))
        categories = case.get("categories", []) or [case.get("category", "general")]

        try:
            issues = check_v4(text)["errors"]
        except Exception as e:
            print(f"  ERROR on case {case.get('id', i)}: {e}")
            issues = []

        system_flagged = len(issues) > 0
        result = ("TP" if should_flag and system_flagged
                  else "FN" if should_flag and not system_flagged
                  else "FP" if not should_flag and system_flagged
                  else "TN")
        results.append({
            "id": case.get("id", i), "text": text, "should_flag": should_flag,
            "system_flagged": system_flagged, "result": result,
            "num_issues": len(issues),
            "issues": [{"rule": e.get("rule_id"), "message": e.get("message", "")[:80],
                        "sources": e.get("sources", [])} for e in issues[:4]],
        })

        for cat in categories:
            s = by_cat[cat]
            s["tp" if result == "TP" else "fp" if result == "FP"
              else "fn" if result == "FN" else "tn"] += 1

        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(cases)}")

    tp = sum(1 for r in results if r["result"] == "TP")
    fp = sum(1 for r in results if r["result"] == "FP")
    fn = sum(1 for r in results if r["result"] == "FN")
    tn = sum(1 for r in results if r["result"] == "TN")
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    fpr = fp / max(fp + tn, 1)

    print("=" * 72)
    print(f"OVERALL  TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}  FPR={fpr:.3f}")
    print("=" * 72)

    print("\nPER-CATEGORY")
    for cat in sorted(by_cat):
        s = by_cat[cat]
        p = s["tp"] / max(s["tp"] + s["fp"], 1)
        r = s["tp"] / max(s["tp"] + s["fn"], 1)
        print(f"  {cat:22s} P={p:.2f} R={r:.2f} FP={s['fp']} FN={s['fn']}")

    fps = [r for r in results if r["result"] == "FP"]
    fns = [r for r in results if r["result"] == "FN"]

    if fps:
        print(f"\nFALSE POSITIVES ({len(fps)})")
        for r in fps:
            print(f"  [{r['id']}] {r['text']}")
            for iss in r["issues"]:
                print(f"      -> {iss['rule']}: {iss['message']} src={iss['sources']}")

    if fns:
        print(f"\nFALSE NEGATIVES ({len(fns)})")
        for r in fns:
            print(f"  [{r['id']}] {r['text']}")

    report = {
        "cases_path": cases_path, "total": len(cases),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4), "fpr": round(fpr, 4),
        "by_category": dict(by_cat),
        "fps": [r["id"] for r in fps], "fns": [r["id"] for r in fns],
        "details": results,
    }
    out = os.path.join(HERE, "v4_benchmark_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {out}")
    return report


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CASES
    run(path)