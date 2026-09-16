import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import get_nlp, NLPDetector
nlp = get_nlp()
det = NLPDetector()

tests = [
    ("everyone were waiting", "SVA: everyone was"),
    ("everybody are waiting", "SVA: everybody is"),
    ("they has spent", "SVA: they have"),
    ("customers has reported", "SVA: customers have"),
    ("food were served", "SVA: food was"),
    ("noodles was too spicy", "SVA: noodles were"),
    ("they uploads large files", "SVA: they upload"),
    ("didn't answered the phone", "DIDNT_PAST: didn't answer"),
    ("didn't came back", "DIDNT_PAST: didn't come"),
    ("didn't had enough", "DIDNT_PAST: didn't have"),
    ("may takes another few weeks", "MODAL: may take"),
    ("will becomes very popular", "MODAL: will become"),
    ("had buy the wrong type", "PP: had bought"),
    ("had forgot my wallet", "PP: had forgotten"),
    ("grandparents house", "POSSESS: grandparents' house"),
    ("a websites ranking", "POSSESS: website's ranking"),
    ("to brings some vegetables", "BASE: to bring"),
    ("search engines looks", "SVA: engines look"),
    ("content are not enough", "SVA: content is"),
]

for text, expected in tests:
    doc = nlp(text)
    errors = det.detect(text, doc)
    if errors:
        found = ", ".join(f"{e.rule_id}:{e.original}->{e.replacement}" for e in errors)
        print(f"  DETECTED: '{text}' -> {found}")
    else:
        print(f"  MISSED:   '{text}' (expected: {expected})")
        # Debug parse
        for tok in doc:
            if tok.dep_ in ("nsubj", "ROOT", "aux", "dobj", "ccomp", "conj", "xcomp"):
                print(f"            PARSE: {tok.text:15s} tag={tok.tag_:4s} dep={tok.dep_:12s} head={tok.head.text}")
