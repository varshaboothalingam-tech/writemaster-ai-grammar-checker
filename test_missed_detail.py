import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import get_nlp, NLPDetector
nlp = get_nlp()
det = NLPDetector()

# S48: "they was angry" in advcl
s48 = "She was making so much noise that they was angry."
doc = nlp(s48)
print("S48 full parse:")
for tok in doc:
    print(f"  {tok.text:12s} tag={tok.tag_:4s} dep={tok.dep_:12s} head={tok.head.text}")
errors = det.detect(s48, doc)
print(f"  Errors: {[e.rule_id + ':' + e.original + '->' + e.replacement for e in errors]}")

print()

# S32: "is too many" in existential
s32 = "There is too many images on the page."
doc = nlp(s32)
print("S32 full parse:")
for tok in doc:
    print(f"  {tok.text:12s} tag={tok.tag_:4s} dep={tok.dep_:12s} head={tok.head.text}")
errors = det.detect(s32, doc)
print(f"  Errors: {[e.rule_id + ':' + e.original + '->' + e.replacement for e in errors]}")

print()

# S35: "have checked" + "yesterday"
s35 = "I have already checked the reports yesterday."
doc = nlp(s35)
print("S35 full parse:")
for tok in doc:
    print(f"  {tok.text:12s} tag={tok.tag_:4s} dep={tok.dep_:12s} head={tok.head.text}")
errors = det.detect(s35, doc)
print(f"  Errors: {[e.rule_id + ':' + e.original + '->' + e.replacement for e in errors]}")

print()

# S44: "studied...will pass"
s44 = "If I studied harder, I will pass the exam."
doc = nlp(s44)
print("S44 full parse:")
for tok in doc:
    print(f"  {tok.text:12s} tag={tok.tag_:4s} dep={tok.dep_:12s} head={tok.head.text}")
errors = det.detect(s44, doc)
print(f"  Errors: {[e.rule_id + ':' + e.original + '->' + e.replacement for e in errors]}")

print()

# S46: "didn't expected"
s46 = "She didn't expected to see him there."
doc = nlp(s46)
print("S46 full parse:")
for tok in doc:
    print(f"  {tok.text:12s} tag={tok.tag_:4s} dep={tok.dep_:12s} head={tok.head.text}")
errors = det.detect(s46, doc)
print(f"  Errors: {[e.rule_id + ':' + e.original + '->' + e.replacement for e in errors]}")

print()

# S47: "every days"
s47 = "Every days I wake up early and go to the gym."
doc = nlp(s47)
print("S47 full parse:")
for tok in doc:
    print(f"  {tok.text:12s} tag={tok.tag_:4s} dep={tok.dep_:12s} head={tok.head.text}")
errors = det.detect(s47, doc)
print(f"  Errors: {[e.rule_id + ':' + e.original + '->' + e.replacement for e in errors]}")

print()

# S28: "who uses" + "have increased"
s28 = "The number of people who uses this application have increased significantly."
doc = nlp(s28)
errors = det.detect(s28, doc)
print(f"S28 Errors: {[e.rule_id + ':' + e.original + '->' + e.replacement for e in errors]}")
