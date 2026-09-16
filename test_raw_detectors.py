"""Raw detector output without filtering."""
import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline, get_nlp, NLPDetector, FastDetector, DataDrivenDetector

p = UnifiedPipeline()
nd = NLPDetector()
fd = FastDetector()
dd = DataDrivenDetector()
nlp = get_nlp()

tests = [
    ("didnt_past", "I did not started the car."),
    ("double_subj", "My cousin she was telling me."),
    ("progressive", "She going to sell it."),
    ("possessive", "The brother number is here."),
    ("plural", "I have many problem."),
    ("regression", "Yesterday my friend and me goes to a large stopping mall because we waited to buy some new clothe."),
]

for name, text in tests:
    doc = nlp(text)
    fast = fd.detect(text)
    nlp_cands = nd.detect(text, doc)
    data_cands = dd.detect(text, doc)
    all_cands = fast + nlp_cands + data_cands
    print(f"--- {name}: {text[:60]} ---")
    for c in all_cands:
        print(f"  [{c.rule_id:25s}] '{c.original}' -> '{c.replacement}' (conf={c.raw_confidence:.2f} det={c.detector})")
    if not all_cands:
        print("  NOTHING")
    print()
