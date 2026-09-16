import json
from collections import Counter

with open(r'C:\Users\hemal\OneDrive\Documents\grammar_checker\tests\grammar_accuracy\benchmark_results.json') as f:
    d = json.load(f)

fns = [x for x in d['details'] if x['should_flag'] and not x['system_flagged']]
print(f'FNs: {len(fns)}')
for fn in fns[:5]:
    print(f"  [{fn['categories']}] {fn['text'][:120]}")

cats = Counter()
for fn in fns:
    for c in fn['categories']:
        cats[c] += 1
print(f'\nFN categories:')
for cat, cnt in cats.most_common(30):
    print(f'  {cat}: {cnt}')

# Show examples per top category
print(f'\n--- FN examples per top category ---')
seen_cats = set()
for fn in fns:
    for c in fn['categories']:
        if c not in seen_cats and cats[c] >= 5:
            seen_cats.add(c)
            print(f'\n  [{c}] ({cats[c]} FNs)')
            print(f"    \"{fn['text'][:140]}\"")
            print(f"    expected: {fn['categories']}")
            if fn.get('issues'):
                for iss in fn['issues'][:2]:
                    print(f"    issue: {iss}")
