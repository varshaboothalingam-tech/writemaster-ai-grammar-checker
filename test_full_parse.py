import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import get_nlp

nlp = get_nlp()

full = """Our company has recently launched a new software product which is designed for helping small businesses manage their customer data. The development team have worked on it since almost one year, and they has spent many hours testing different features. Although the product looks very promising, there are still several problems which needs to be fixed before it can be released to the public. Some customers has reported that the application loads very slowly when they uploads large files. The developers are currently working on solving these issues, but they said that it may takes another few weeks. The marketing team have already prepared several articles and advertisements, so everybody are waiting for the final release. If the technical problems will be solved soon, the company expects that the product will becomes very popular among small businesses."""

doc = nlp(full)
for sent in doc.sents:
    text = sent.text
    if any(w in text for w in ["problems", "customers", "has", "takes", "becomes", "are waiting", "have already"]):
        print(f"\nSENT: '{text}'")
        for token in sent:
            print(f"  {token.text:15s} tag={token.tag_:5s} dep={token.dep_:10s} head={token.head.text}")
