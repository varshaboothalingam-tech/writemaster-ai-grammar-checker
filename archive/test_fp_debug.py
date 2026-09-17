import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline, get_nlp, NLPDetector, FalsePositiveFilter
from unified_pipeline import ErrorCandidate, ErrorCategory
nlp = get_nlp()
det = NLPDetector()
fpf = FalsePositiveFilter()

s1 = "My cousin she was telling me about her new car and she also say that she going to sell her old car because it have many problem."
s2 = "When I ask her why she selling it, she tell me because the car was too boring and she dont like it."

for i, s in enumerate([s1, s2], 1):
    doc = nlp(s)
    raw = det.detect(s, doc)
    print(f"Sentence {i}: {s[:70]}...")
    print(f"  Raw errors: {len(raw)}")
    for e in raw:
        print(f"    {e.rule_id:35s} orig='{e.original}' repl='{e.replacement}'")
    
    # Now apply FP filter
    filtered = fpf.filter(raw, s, doc)
    print(f"  After FP filter: {len(filtered)} (filtered out {len(raw) - len(filtered)})")
    for e in filtered:
        print(f"    {e.rule_id:35s} orig='{e.original}' repl='{e.replacement}'")
    
    # Check what got filtered and why
    filtered_ids = {e.rule_id for e in filtered}
    for e in raw:
        if e.rule_id not in filtered_ids:
            print(f"  FILTERED: {e.rule_id:35s} orig='{e.original}' repl='{e.replacement}'")
    print()
