import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import get_nlp
nlp = get_nlp()

s = "She explained me the process."
doc = nlp(s)
for tok in doc:
    print(f"  {tok.text:12s} tag={tok.tag_:4s} dep={tok.dep_:12s} head={tok.head.text}")

s2 = "I prefer coffee than tea."
doc2 = nlp(s2)
print("\nprefer parse:")
for tok in doc2:
    print(f"  {tok.text:12s} tag={tok.tag_:4s} dep={tok.dep_:12s} head={tok.head.text}")
