"""Debug user text - raw detector output."""
import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import get_nlp, NLPDetector
nlp = get_nlp()
det = NLPDetector()

s1 = "My cousin she was telling me about her new car and she also say that she going to sell her old car because it have many problem."
s2 = "When I ask her why she selling it, she tell me because the car was too boring and she dont like it."

for i, s in enumerate([s1, s2], 1):
    doc = nlp(s)
    errors = det.detect(s, doc)
    print(f"Sentence {i}: {s[:70]}...")
    for e in errors:
        print(f"  {e.rule_id:35s} '{e.original}' -> '{e.replacement}' conf={e.raw_confidence:.2f}")
    print()
