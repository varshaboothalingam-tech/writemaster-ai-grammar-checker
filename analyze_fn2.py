import json
with open(r'C:\Users\hemal\OneDrive\Documents\grammar_checker\tests\grammar_accuracy\benchmark_results.json') as f:
    data = json.load(f)

# Show FN examples for top categories
for cat in ['grammar', 'punctuation', 'sentence_structure', 'tense', 'word_usage', 'spelling', 'agreement', 'prepositions', 'pronouns']:
    fns = [d for d in data['details'] if d['result'] == 'FN' and cat in d['categories']]
    print(f"\n=== {cat.upper()} ({len(fns)} FNs) ===")
    for d in fns[:8]:
        print(f"  [{d['id']}] {d['text'][:90]}")
