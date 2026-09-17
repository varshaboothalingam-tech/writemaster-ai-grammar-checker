import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import *

p = UnifiedPipeline()
nlp = get_nlp()

# Test each detector individually
tests = {
    'since_for': "I haven't visited them since two years.",
    'one_of': "One of my cousin suggested that we takes a shortcut.",
    'article_a_an': "He is a MBA graduate.",
    'tense': "She walked to the store and buys some milk.",
    'possessive': "The company released it's quarterly report.",
    'obj_pronoun': "Give the message to him and I.",
    'parallelism': "She likes to swim, running, and to bike.",
}

for name, text in tests.items():
    doc = nlp(text)
    nd = NLPDetector()
    print(f'\n--- {name}: "{text}" ---')
    all_cands = nd.detect(text, doc)
    for c in all_cands:
        print(f'  [{c.rule_id}] "{c.original}" -> "{c.replacement}" conf={c.raw_confidence:.2f}')
    if not all_cands:
        print('  NO CANDIDATES!')
