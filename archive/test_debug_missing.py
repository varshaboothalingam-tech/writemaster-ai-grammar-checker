"""Debug specific missing detections."""
import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import get_nlp, NLPDetector
nlp = get_nlp()
det = NLPDetector()

tests = [
    "it didn't started properly",
    "since two years",
    "she was not satisfy",
    "she was also worry",
    "the brakes was not working",
    "who want to buy",
    "my friend have a brother",
    "I tell her",
    "the car keep break down",
    "lot of money",
]
for t in tests:
    doc = nlp(t)
    errors = det.detect(t, doc)
    print(f"Text: {t}")
    if errors:
        for e in errors:
            print(f"  {e.rule_id:35s} '{e.original}' -> '{e.replacement}'")
    else:
        print(f"  (no errors detected)")
    print()
