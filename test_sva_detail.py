"""Debug SVA for specific patterns."""
import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import get_nlp
nlp = get_nlp()

# Check what SVA detects on individual sentences
s1 = "the brakes was not working properly"
s2 = "who want to buy a cheap car"
s3 = "my friend have a brother"
s4 = "She was not satisfy with the car"

for t in [s1, s2, s3, s4]:
    doc = nlp(t)
    print(f"\n'{t}'")
    for tok in doc:
        if tok.dep_ in ("nsubj", "nsubjpass", "ROOT", "aux"):
            print(f"  {tok.text:15s} tag={tok.tag_:4s} dep={tok.dep_:12s} head={tok.head.text} pos={tok.pos_}")
