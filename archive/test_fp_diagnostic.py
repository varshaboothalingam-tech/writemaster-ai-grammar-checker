"""Quick diagnostic: show rule_ids for FP sentences."""
import json, sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline

pipeline = UnifiedPipeline()

with open(r"C:\Users\hemal\OneDrive\Documents\grammar_checker\tests\grammar_accuracy\benchmark_1000.json", "r", encoding="utf-8") as f:
    cases = json.load(f)

fp_count = 0
for case in cases:
    if case.get("should_flag", False):
        continue
    issues = pipeline.check(case["text"])
    if issues:
        fp_count += 1
        rule_ids = [i.get("rule_id", "?") for i in issues]
        cats = case.get("categories", [])
        print(f"  [{case.get('id', '?')}] {case['text'][:80]}")
        print(f"    cats={cats} rules={rule_ids}")
        print()

print(f"Total FPs: {fp_count}")
