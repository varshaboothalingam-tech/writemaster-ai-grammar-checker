import json
from collections import Counter

with open(r'C:\Users\hemal\OneDrive\Documents\grammar_checker\tests\grammar_accuracy\benchmark_results.json') as f:
    d = json.load(f)

fns = [x for x in d['details'] if x['should_flag'] and not x['system_flagged']]
print(f'Total FNs: {len(fns)}')

# Count FN categories
cats = Counter()
for fn in fns:
    for c in fn['categories']:
        cats[c] += 1

print('\nFN by category:')
for cat, cnt in cats.most_common():
    print(f'  {cat}: {cnt}')

# Show grammar FNs specifically
print('\n=== GRAMMAR FNs (sample) ===')
grammar_fns = [fn for fn in fns if 'grammar' in fn['categories']]
for fn in grammar_fns[:30]:
    print(f"  \"{fn['text'][:100]}\"")

# Show punctuation FNs
print('\n=== PUNCTUATION FNs (sample) ===')
punct_fns = [fn for fn in fns if 'punctuation' in fn['categories']]
for fn in punct_fns[:15]:
    print(f"  \"{fn['text'][:100]}\"")

# Show sentence_structure FNs
print('\n=== SENTENCE STRUCTURE FNs (sample) ===')
ss_fns = [fn for fn in fns if 'sentence_structure' in fn['categories']]
for fn in ss_fns[:15]:
    print(f"  \"{fn['text'][:100]}\"")

# Show tense FNs
print('\n=== TENSE FNs (sample) ===')
tense_fns = [fn for fn in fns if 'tense' in fn['categories']]
for fn in tense_fns[:15]:
    print(f"  \"{fn['text'][:100]}\"")

# Show preposition FNs
print('\n=== PREPOSITION FNs (sample) ===')
prep_fns = [fn for fn in fns if 'prepositions' in fn['categories']]
for fn in prep_fns[:15]:
    print(f"  \"{fn['text'][:100]}\"")

# Show word_usage FNs
print('\n=== WORD USAGE FNs (sample) ===')
wu_fns = [fn for fn in fns if 'word_usage' in fn['categories']]
for fn in wu_fns[:15]:
    print(f"  \"{fn['text'][:100]}\"")
