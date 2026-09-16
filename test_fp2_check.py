import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline
p = UnifiedPipeline()

tests = [
    "We shouldn't be doing this.",
    "We'd been waiting for two hours.",
    "Not until the sun went down did they notice the fire.",
    "We crossed that bridge when we came to it.",
    "I wouldn't do that if I were you.",
]
for t in tests:
    r = p.check(t)
    if r:
        print(f"  FP: \"{t}\" => {[(e['original'],e['replacement'],e['rule_id']) for e in r]}")
    else:
        print(f"  OK: \"{t}\"")
