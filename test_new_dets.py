import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import get_nlp, NLPDetector
nlp = get_nlp()
det = NLPDetector()

tests = [
    "She explained me the process.",
    "He suggested me to apply.",
    "I prefer coffee than tea.",
]
for s in tests:
    doc = nlp(s)
    errors = det.detect(s, doc)
    print(f"'{s}'")
    for e in errors:
        print(f"  {e.rule_id:20s} [{e.start}:{e.end}] '{e.original}' -> '{e.replacement}'")
    print()
