import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import get_nlp, NLPDetector

nlp = get_nlp()
det = NLPDetector()

tests = [
    "they was celebrating",
    "we was expecting it",
    "food were served after twenty minutes",
    "team have worked on it",
    "problems which needs fixing",
    "competitors websites are important",
]
for t in tests:
    doc = nlp(t)
    print(f"\n=== '{t}' ===")
    for tok in doc:
        print(f"  {tok.text:15s} tag={tok.tag_:5s} dep={tok.dep_:10s} head={tok.head.text}")
    errs = det.detect(t, doc)
    for e in errs:
        print(f"  DETECTED: {e.rule_id}: '{e.original}' -> '{e.replacement}'")
    if not errs:
        print("  NO ERRORS FOUND")
