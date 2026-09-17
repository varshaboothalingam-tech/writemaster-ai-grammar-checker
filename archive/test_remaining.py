import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline, NLPDetector, get_nlp

p = UnifiedPipeline()
nlp = get_nlp()
det = NLPDetector()

# Sentences that work individually but not in pipeline
tests = [
    (46, "I didn't expected the meeting to take so long, so I haven't brought anything to eat."),
    (47, "My sister bought a new laptop because her old one was becoming slower and slower every days."),
    (48, "The teacher asked the students why they was making so much noise during the examination."),
    (35, "By the time we arrived at the hotel, our friends have already checked in and went to their rooms."),
    (40, "We should focus on improving the quality of the content rather than to publish more articles every week."),
]

for num, s in tests:
    doc = nlp(s)
    nlp_errors = det.detect(s, doc)
    pipe_errors = p.check(s)
    print(f"S{num}:")
    if nlp_errors:
        for e in nlp_errors:
            print(f"  NLP: {e.rule_id:30s} [{e.start}:{e.end}] '{e.original}' -> '{e.replacement}'")
    else:
        print(f"  NLP: 0 issues")
    if pipe_errors:
        for e in pipe_errors:
            print(f"  PIPE: {e['rule_id']:30s} '{e['original']}' -> '{e['replacement']}'")
    else:
        print(f"  PIPE: 0 issues")
    print()
