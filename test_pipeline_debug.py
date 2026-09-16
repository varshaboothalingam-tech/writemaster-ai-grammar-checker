import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline, NLPDetector, get_nlp

p = UnifiedPipeline()
nlp = get_nlp()
det = NLPDetector()

problem_sentences = [
    (28, "The number of people who uses this application have increased significantly during the last few months."),
    (29, "Every students in the class were asked to submit their assignments before the end of the week."),
    (39, "He did not told anyone about the problem because he was afraid that his manager might gets angry."),
    (47, "The customer complained that nobody have responded to his email for more than three days."),
    (48, "He told me that he did not understood the instructions because they were written in a very complicated way."),
    (50, "If she studied harder, she would have passed the examination last month."),
]

for num, s in problem_sentences:
    doc = nlp(s)
    nlp_errors = det.detect(s, doc)
    pipe_errors = p.check(s)
    
    print(f"S{num}: '{s[:70]}...'")
    print(f"  NLP raw: {len(nlp_errors)} issues")
    for e in nlp_errors:
        print(f"    {e.rule_id:30s} '{e.original}' -> '{e.replacement}' conf={e.raw_confidence:.2f}")
    print(f"  Pipeline: {len(pipe_errors)} issues")
    for e in pipe_errors:
        print(f"    {e['rule_id']:30s} '{e['original']}' -> '{e['replacement']}' conf={e['confidence']:.2f}")
    print()
