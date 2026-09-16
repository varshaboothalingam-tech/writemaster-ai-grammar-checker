import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import get_nlp, NLPDetector
nlp = get_nlp()
det = NLPDetector()

# Test: full text sentence
s = "Some customers has reported that the application loads very slowly when they uploads large files."
doc = nlp(s)
errors = det.detect(s, doc)
print(f"Individual: {len(errors)} errors")
for e in errors:
    print(f"  {e.rule_id:25s} '{e.original}' -> '{e.replacement}'")

# Now test with preceding text (mimics full text context)
full_s = "Our company has recently launched a new software product. Some customers has reported that the application loads very slowly when they uploads large files."
doc2 = nlp(full_s)
errors2 = det.detect(full_s, doc2)
print(f"\nWith context: {len(errors2)} errors")
for e in errors2:
    print(f"  {e.rule_id:25s} [{e.start}:{e.end}] '{e.original}' -> '{e.replacement}'")

# Check the parse difference
print("\nParse of 'customers has reported':")
for sent in doc2.sents:
    if "customers" in sent.text:
        for tok in sent:
            if tok.text == "customers" or tok.text == "has" or tok.text == "reported":
                print(f"  {tok.text:12s} tag={tok.tag_:4s} dep={tok.dep_:12s} head={tok.head.text}")
        # Check what SVA detector would see
        for tok in sent:
            if tok.dep_ == "nsubj" and tok.text == "customers":
                verb = tok.head
                print(f"\n  nsubj='customers' -> verb='{verb.text}' ({verb.tag_}, {verb.dep_})")
                # Check if verb has aux
                for child in verb.children:
                    if child.dep_ == "aux":
                        print(f"    aux: '{child.text}' ({child.tag_})")
                # Check _get_expected_verb
                expected = det._get_expected_verb(verb, "plural", sent)
                print(f"    _get_expected_verb(verb, 'plural') = {expected}")
                current = verb.text.lower()
                print(f"    current='{current}' expected='{expected}' match={current == expected}")
