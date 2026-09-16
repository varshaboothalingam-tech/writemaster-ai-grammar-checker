import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline, NLPDetector, get_nlp

det = NLPDetector()
nlp = get_nlp()

s32 = "The website loads very slowly because there is too many images and unnecessary scripts on the page."
doc = nlp(s32)
errors = det.detect(s32, doc)
print(f"S32 NLP: {len(errors)} errors")
for e in errors:
    print(f"  {e.rule_id} [{e.start}:{e.end}] '{e.original}' -> '{e.replacement}' subj={e.metadata.get('subject','?')}")

# Check why EXISTENTIAL_THERE_SVA doesn't fire
print("\nParse:")
for tok in doc:
    if tok.dep_ in ("expl", "attr", "ROOT"):
        print(f"  {tok.text:12s} tag={tok.tag_:4s} dep={tok.dep_:12s} head={tok.head.text}")

# Check tense issue
s35 = "By the time we arrived at the hotel, our friends have already checked in and went to their rooms."
doc = nlp(s35)
errors = det.detect(s35, doc)
print(f"\nS35 NLP: {len(errors)} errors")
for e in errors:
    print(f"  {e.rule_id} [{e.start}:{e.end}] '{e.original}' -> '{e.replacement}'")
print("Parse:")
for tok in doc:
    if tok.dep_ in ("aux", "ROOT", "conj", "nsubj"):
        print(f"  {tok.text:12s} tag={tok.tag_:4s} dep={tok.dep_:12s} head={tok.head.text}")
