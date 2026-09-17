import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import get_nlp, NLPDetector
nlp = get_nlp()
det = NLPDetector()

# Sentences still getting 0 detections
sentences = {
    15: "Although it was raining but we went outside.",
    17: "The teacher explained me how does the system works.",
    20: "The manager give us many works to complete.",
    28: "The number of people who uses this application have increased significantly.",
    29: "I prefer coffee than tea.",
    32: "There is too many images on the page.",
    35: "I have already checked the reports yesterday.",
    39: "I don't have none left.",
    40: "The researcher decided to continue rather than to publish.",
    44: "If I studied harder, I will pass the exam.",
    46: "She didn't expected to see him there.",
    47: "Every days I wake up early and go to the gym.",
    48: "She was making so much noise that they was angry.",
}

for num, s in sorted(sentences.items()):
    doc = nlp(s)
    errors = det.detect(s, doc)
    if not errors:
        # Debug parse
        print(f"S{num}: '{s}'")
        for tok in doc:
            if tok.dep_ in ("nsubj", "ROOT", "relcl", "ccomp", "advcl", "dobj", "conj"):
                print(f"  {tok.text:15s} tag={tok.tag_:4s} dep={tok.dep_:12s} head={tok.head.text}")
        print()
