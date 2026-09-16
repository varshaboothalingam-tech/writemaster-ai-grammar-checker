import json

with open('C:/Users/hemal/OneDrive/Documents/grammar_checker/tests/grammar_accuracy/benchmark_results.json') as f:
    data = json.load(f)

fns_by_cat = {}
for r in data['results']:
    if not r['is_correct']:
        errors_expected = [e for e in r['expected_errors'] if not e.get('found', False)]
        for e in errors_expected:
            cat = e.get('category', 'unknown')
            if cat not in fns_by_cat:
                fns_by_cat[cat] = []
            fns_by_cat[cat].append({
                'text': r['text'][:100],
                'expected': e.get('expected_correction', ''),
                'error_type': e.get('error_type', ''),
                'rule_id': e.get('rule_id', ''),
            })

for cat in sorted(fns_by_cat.keys(), key=lambda x: -len(fns_by_cat[x])):
    items = fns_by_cat[cat]
    print(f'\n=== {cat.upper()} ({len(items)} FNs) ===')
    for item in items[:8]:
        print(f'  "{item["text"]}"')
        print(f'    Expected: {item["expected"]}')
        print(f'    Error: {item["error_type"]} | Rule: {item["rule_id"]}')
    if len(items) > 8:
        print(f'  ... and {len(items)-8} more')
