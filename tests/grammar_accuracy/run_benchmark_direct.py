"""Direct benchmark — no server needed."""
import json
import sys
import time
from collections import defaultdict

sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline

BENCHMARK_PATH = r"C:\Users\hemal\OneDrive\Documents\grammar_checker\tests\grammar_accuracy\benchmark_1000.json"

pipeline = UnifiedPipeline()

with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
    cases = json.load(f)

print(f"Running direct benchmark: {len(cases)} sentences")
print("=" * 60)

results = []
errors_by_category = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})

start_time = time.time()

for i, case in enumerate(cases):
    text = case["text"]
    should_flag = case.get("should_flag", False)
    categories = case.get("categories", [])

    issues = pipeline.check(text)
    system_flagged = len(issues) > 0

    if should_flag and system_flagged:
        r = "TP"
    elif should_flag and not system_flagged:
        r = "FN"
    elif not should_flag and system_flagged:
        r = "FP"
    else:
        r = "TN"

    results.append({"id": case.get("id", i), "result": r, "categories": categories,
                     "num_issues": len(issues), "text": text[:60]})

    for cat in categories:
        if r == "TP": errors_by_category[cat]["tp"] += 1
        elif r == "FN": errors_by_category[cat]["fn"] += 1
        elif r == "FP": errors_by_category[cat]["fp"] += 1
        else: errors_by_category[cat]["tn"] += 1

    if (i + 1) % 100 == 0:
        elapsed = time.time() - start_time
        print(f"  Processed {i + 1}/{len(cases)}... ({elapsed:.1f}s)")

elapsed = time.time() - start_time

tp = sum(1 for r in results if r["result"] == "TP")
fp = sum(1 for r in results if r["result"] == "FP")
fn = sum(1 for r in results if r["result"] == "FN")
tn = sum(1 for r in results if r["result"] == "TN")

precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
f1 = 2 * precision * recall / max(precision + recall, 0.001)
fpr = fp / max(fp + tn, 1)

print(f"\n{'='*60}")
print(f"OVERALL RESULTS ({elapsed:.1f}s)")
print(f"{'='*60}")
print(f"  True Positives:  {tp}")
print(f"  False Positives: {fp}")
print(f"  False Negatives: {fn}")
print(f"  True Negatives:  {tn}")
print(f"  Precision:       {precision:.3f} ({precision*100:.1f}%)")
print(f"  Recall:          {recall:.3f} ({recall*100:.1f}%)")
print(f"  F1 Score:        {f1:.3f}")
print(f"  False Pos Rate:  {fpr:.3f} ({fpr*100:.1f}%)")

# Per-category
print(f"\n{'='*60}")
print("PER-CATEGORY RESULTS")
print(f"{'='*60}")
for cat in sorted(errors_by_category.keys()):
    d = errors_by_category[cat]
    p = d["tp"] / max(d["tp"] + d["fp"], 1)
    r = d["tp"] / max(d["tp"] + d["fn"], 1)
    f = 2 * p * r / max(p + r, 0.001)
    if d["tp"] + d["fp"] + d["fn"] > 0:
        print(f"  {cat:30s} P={p:.2f} R={r:.2f} F1={f:.2f} FP={d['fp']} FN={d['fn']}")

# Print FPs
print(f"\n{'='*60}")
print(f"FALSE POSITIVES ({fp})")
print(f"{'='*60}")
for r in results:
    if r["result"] == "FP":
        print(f"  [{r['id']}] {r['text']}  (cats={r['categories']})")

# Save
with open(r"C:\Users\hemal\OneDrive\Documents\grammar_checker\tests\grammar_accuracy\benchmark_results.json", "w") as f:
    json.dump({"tp": tp, "fp": fp, "fn": fn, "tn": tn,
               "precision": precision, "recall": recall, "f1": f1, "fpr": fpr,
               "results": results}, f, indent=2)
