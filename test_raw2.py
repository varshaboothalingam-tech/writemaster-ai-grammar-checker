import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import get_nlp, NLPDetector
nlp = get_nlp()
det = NLPDetector()
s = 'My cousin she was telling me about her new car and she also say that she going to sell her old car because it have many problem.'
doc = nlp(s)
errors = det.detect(s, doc)
print('All raw errors:')
for e in errors:
    print(f'  {e.rule_id:35s} pos={e.start:3d} orig="{e.original}" repl="{e.replacement}" conf={e.raw_confidence:.2f}')
print()
print('MISSING_AUXILIARY:', [e.rule_id for e in errors if e.rule_id == 'MISSING_AUXILIARY'])
print('MISSING_PLURAL:', [e.rule_id for e in errors if e.rule_id == 'MISSING_PLURAL'])
