import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline, get_nlp, NLPDetector

p = UnifiedPipeline()
nlp = get_nlp()
det = NLPDetector()

s = "My sister bought a new laptop because her old one was becoming slower and slower every days."
doc = nlp(s)
nlp_errors = det.detect(s, doc)
for e in nlp_errors:
    print(f"NLP: {e.rule_id} [{e.start}:{e.end}] '{e.original}' is_fp={p.fp_filter._is_false_positive(e, s, doc)}")
    # Check entity
    for ent in doc.ents:
        if ent.start_char <= e.start < ent.end_char:
            print(f"  Inside entity: '{ent.text}' [{ent.start_char}:{ent.end_char}] label={ent.label_}")
    print(f"  subject={e.metadata.get('subject','?')}")

pipe_errors = p.check(s)
for e in pipe_errors:
    print(f"PIPE: {e['rule_id']} '{e['original']}' -> '{e['replacement']}'")
