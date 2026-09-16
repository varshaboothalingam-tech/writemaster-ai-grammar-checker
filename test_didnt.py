import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import get_nlp, NLPDetector
nlp = get_nlp()
det = NLPDetector()

s7 = "He asked me where was I going, but I didn't knew how to explain the situation properly."
s42 = "He told me that he didn't understood the instructions because they were written in a very complicated way."
s46 = "I didn't expected the meeting to take so long, so I haven't brought anything to eat."

for s in [s7, s42, s46]:
    doc = nlp(s)
    errors = det.detect(s, doc)
    print(f"'{s[:60]}...'")
    for e in errors:
        print(f"  {e.rule_id:30s} '{e.original}' -> '{e.replacement}'")
    if not errors:
        print(f"  (none)")
    # Check parse of the key phrase
    for tok in doc:
        if tok.text.lower() in ("knew", "understood", "expected"):
            print(f"  PARSE: {tok.text} tag={tok.tag_} dep={tok.dep_} head={tok.head.text}")
    print()
