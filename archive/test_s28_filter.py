import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline, NLPDetector, get_nlp
from unified_pipeline import FalsePositiveFilter

p = UnifiedPipeline()
nlp = get_nlp()
det = NLPDetector()
fpf = FalsePositiveFilter()

s = "The number of people who uses this application have increased significantly during the last few months."
doc = nlp(s)

# Step 1: Raw NLP
nlp_errors = det.detect(s, doc)
print(f"NLP raw: {len(nlp_errors)} issues")
for e in nlp_errors:
    print(f"  {e.rule_id:30s} [{e.start}:{e.end}] '{e.original}' -> '{e.replacement}'")

# Step 2: FP filter
filtered = fpf.filter_errors(nlp_errors, s)
print(f"\nAfter FP filter: {len(filtered)} issues")
for e in filtered:
    print(f"  {e.rule_id:30s} [{e.start}:{e.end}] '{e.original}' -> '{e.replacement}'")

# Step 3: Full pipeline
pipe_errors = p.check(s)
print(f"\nPipeline: {len(pipe_errors)} issues")
for e in pipe_errors:
    print(f"  {e['rule_id']:30s} '{e['original']}' -> '{e['replacement']}'")

# Check dedup
print(f"\nDedup check:")
for e in nlp_errors:
    matched = p.check(s)
    # Check if the error is being deduplicated
    for pe in matched:
        if pe['start'] == e.start and pe['end'] == e.end:
            print(f"  MATCHED: {e.rule_id} at {e.start}:{e.end}")
            break
    else:
        print(f"  DROPPED: {e.rule_id} at {e.start}:{e.end} '{e.original}' -> '{e.replacement}'")
