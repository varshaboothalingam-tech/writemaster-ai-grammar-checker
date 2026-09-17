import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import get_nlp

nlp = get_nlp()
text = "there are still several problems which needs to be fixed"
doc = nlp(text)
for token in doc:
    print(f"  {token.text:15s} tag={token.tag_:5s} pos={token.pos_:5s} dep={token.dep_:10s} head={token.head.text}")
