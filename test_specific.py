"""Test specific detectors for missing patterns."""
import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline, get_nlp, NLPDetector

p = UnifiedPipeline()
nd = NLPDetector()
nlp = get_nlp()

tests = [
    ("didnt_past", "I did not started the car."),
    ("didnt_past2", "She did not brought her bag."),
    ("double_subj", "My cousin she was telling me."),
    ("progressive", "She going to sell it."),
    ("progressive2", "She selling it."),
    ("possessive", "The brother number is here."),
    ("tense_past", "When I ask her why, she told me."),
    ("plural", "It have many problem."),
]

for name, text in tests:
    doc = nlp(text)
    cands = nd.detect(text, doc)
    filtered = [c for c in cands if c.rule_id]
    if filtered:
        for c in filtered:
            print(f"  [{name}] {c.rule_id}: '{c.original}' -> '{c.replacement}' (conf={c.raw_confidence:.2f})")
    else:
        print(f"  [{name}] NOTHING DETECTED")
