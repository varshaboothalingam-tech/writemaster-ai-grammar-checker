"""Debug specific patterns."""
import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import get_nlp
nlp = get_nlp()

tests = [
    "She was not satisfy with the car",
    "the brakes was not working properly",
    "who want to buy a cheap car",
    "my friend have a brother",
    "she have to spend lot of money",
]
for t in tests:
    doc = nlp(t)
    print(f"\n'{t}'")
    for tok in doc:
        if tok.pos_ in ("NOUN", "PRON", "VERB", "AUX"):
            print(f"  {tok.text:15s} pos={tok.pos_:5s} tag={tok.tag_:4s} dep={tok.dep_:12s} head={tok.head.text}")
