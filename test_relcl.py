import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import get_nlp
nlp = get_nlp()

# Check full sentence parse
s1 = "The company offers software solutions which provide solutions for businesses."
s2 = "The number of people who uses this application have increased significantly."
s3 = "There is too many images on the page."
s4 = "She is one of the best employees that works hard."
for s in [s1, s2, s3, s4]:
    doc = nlp(s)
    print(f"\n'{s[:70]}'")
    for tok in doc:
        if tok.dep_ in ("relcl", "nsubj") or tok.tag_ in ("WP", "WDT"):
            print(f"  {tok.text:12s} tag={tok.tag_:4s} dep={tok.dep_:12s} head={tok.head.text} head_dep={tok.head.dep_}")
