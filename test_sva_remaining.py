import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import get_nlp, NLPDetector
nlp = get_nlp()
det = NLPDetector()

tests = [
    "which provide software solutions",
    "who uses this application",
    "there is too many images",
    "they was making so much noise",
]
for t in tests:
    doc = nlp(t)
    errors = det.detect(t, doc)
    print(f"'{t}'")
    for e in errors:
        print(f"  {e.rule_id:30s} '{e.original}' -> '{e.replacement}'")
    # Parse
    for tok in doc:
        if tok.dep_ in ("nsubj", "ROOT", "relcl", "ccomp", "advcl"):
            print(f"  PARSE: {tok.text} tag={tok.tag_} dep={tok.dep_} head={tok.head.text}")
    print()
