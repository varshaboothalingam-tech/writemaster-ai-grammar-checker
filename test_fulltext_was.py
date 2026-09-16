import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import get_nlp, NLPDetector

nlp = get_nlp()
det = NLPDetector()

text = """Yesterday, my friend and me decided to visit a new restaurant which recently opened near our house. We have heard many good reviews about the food, so we was expecting it to be very good. When we arrived at the restaurant, there were a long queue outside because the place was already crowded."""

doc = nlp(text)
for sent in doc.sents:
    if "was" in sent.text.lower() and "expecting" in sent.text.lower():
        print(f"SENT: '{sent.text}'")
        for tok in sent:
            print(f"  {tok.text:15s} tag={tok.tag_:5s} dep={tok.dep_:10s} head={tok.head.text}")

errs = det.detect(text, doc)
for e in errs:
    ctx = text[max(0, e.start-15):min(len(text), e.end+15)]
    print(f"  {e.rule_id}: '{e.original}' -> '{e.replacement}' ctx='...{ctx}...'")
