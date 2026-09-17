import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline, FalsePositiveFilter, get_nlp, NLPDetector, ErrorCandidate

p = UnifiedPipeline()
s = "A number of students are absent today."

# Simulate pipeline steps
clean_text, protected = p.preprocessor.protect(s)
doc = get_nlp()(clean_text)
nlp_errors = p.nlp_detector.detect(clean_text, doc)
print(f"clean_text: '{clean_text}'")
for e in nlp_errors:
    print(f"  {e.rule_id} start={e.start} end={e.end} subj={e.metadata.get('subject','?')} '{e.original}' -> '{e.replacement}'")
    text_before = clean_text[:e.start].rstrip()
    print(f"  text_before='{text_before}'")

# Manually run FP filter
fpf = p.fp_filter
filtered = fpf.filter(nlp_errors, clean_text, doc)
print(f"\nAfter FP filter: {len(filtered)} (was {len(nlp_errors)})")
for e in filtered:
    print(f"  {e.rule_id} start={e.start} '{e.original}' -> '{e.replacement}'")
