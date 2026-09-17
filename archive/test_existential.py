import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import get_nlp
nlp = get_nlp()

s = "The website loads very slowly because there is too many images and unnecessary scripts on the page."
doc = nlp(s)
print("Full parse:")
for tok in doc:
    print(f"  {tok.text:15s} tag={tok.tag_:4s} pos={tok.pos_:6s} dep={tok.dep_:12s} head={tok.head.text}")

print("\nChecking existential there:")
for tok in doc:
    if tok.lower_ == "there" and tok.dep_ == "expl":
        verb = tok.head
        print(f"  there -> verb: {verb.text} pos={verb.pos_}")
        for child in verb.children:
            print(f"    child: {child.text} dep={child.dep_} pos={child.pos_}")
            if child.dep_ == "attr":
                print(f"    attr: {child.text} tag={child.tag_} pos={child.pos_}")
                plural = nlp.get_pipe_component("ner") if hasattr(nlp, 'get_pipe_component') else None
