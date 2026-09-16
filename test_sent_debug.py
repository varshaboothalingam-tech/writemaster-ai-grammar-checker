"""Why don't these fire in the full text?"""
import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import get_nlp, NLPDetector
nlp = get_nlp()
det = NLPDetector()

# Full text - check individual sentences
sents = [
    "She also say that she was not satisfy with the car because the engine make too much noise and she didn't had enough money to fix it.",
    "My cousin she was very upset because she have to spend lot of money on the car every month.",
    "She was also worry about the safety because the brakes was not working properly and she have a big family who she need to drive to school every day.",
    "She ask me if I know someone who want to buy a cheap car.",
    "I tell her that my friend have a brother who was looking for a car.",
]
for s in sents:
    doc = nlp(s)
    errors = det.detect(s, doc)
    print(f"\n{s[:80]}...")
    for e in errors:
        print(f"  {e.rule_id:30s} '{e.original}' -> '{e.replacement}'")
    if not errors:
        print(f"  (none)")
