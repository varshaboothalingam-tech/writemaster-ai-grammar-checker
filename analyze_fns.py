import json

with open(r'C:\Users\hemal\OneDrive\Documents\grammar_checker\tests\grammar_accuracy\benchmark_results.json') as d:
    data = json.load(d)

with open(r'C:\Users\hemal\OneDrive\Documents\grammar_checker\tests\grammar_accuracy\benchmark_1000.json') as f:
    cases = json.load(f)

case_map = {c.get('id', i): c for i, c in enumerate(cases)}
fns = [r for r in data['results'] if r['result'] == 'FN']

print(f"Total FNs: {len(fns)}\n")

# Sample by category
cats = {}
for fn in fns:
    for cat in fn.get('categories', []):
        cats.setdefault(cat, []).append(fn)

for cat in sorted(cats.keys(), key=lambda c: -len(cats[c])):
    items = cats[cat][:5]
    print(f"\n=== {cat} ({len(cats[cat])} FNs) ===")
    for item in items:
        cid = item['id']
        c = case_map.get(cid, {})
        text = c.get('text', '')[:100]
        expected = c.get('expected_errors', [])
        exp_str = '; '.join([f"{e.get('original','')}->{e.get('correction','')}" for e in expected[:3]])
        print(f"  [{cid}] \"{text}\"")
        print(f"       Expected: {exp_str}")
