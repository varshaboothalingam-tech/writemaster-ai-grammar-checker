import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import get_nlp, NLPDetector
nlp = get_nlp()
det = NLPDetector()

s = "A number of students are absent today."
doc = nlp(s)
for tok in doc:
    print(f"  {tok.text:12s} tag={tok.tag_:4s} dep={tok.dep_:12s} head={tok.head.text}")
print()
errors = det.detect(s, doc)
for e in errors:
    print(f"  {e.rule_id} subj={e.metadata.get('subject','?')} '{e.original}' -> '{e.replacement}'")
