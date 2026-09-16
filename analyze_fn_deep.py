import json, requests

BENCHMARK_PATH = r"C:\Users\hemal\OneDrive\Documents\grammar_checker\tests\grammar_accuracy\benchmark_1000.json"
API_URL = "http://127.0.0.1:5001/api/check-v3"

with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
    cases = json.load(f)

fn_categories = {}
fp_categories = {}

for i, case in enumerate(cases):
    text = case["text"]
    should_flag = case.get("should_flag", False)
    cats = case.get("categories", [])
    
    try:
        resp = requests.post(API_URL, json={"text": text}, timeout=30)
        data = resp.json()
        issues = data.get("issues", [])
    except:
        issues = []
    
    system_flagged = len(issues) > 0
    
    if should_flag and not system_flagged:
        for cat in cats:
            if cat not in fn_categories:
                fn_categories[cat] = []
            fn_categories[cat].append(text[:120])
    elif not should_flag and system_flagged:
        for cat in cats:
            if cat not in fp_categories:
                fp_categories[cat] = []
            fp_categories[cat].append({
                'text': text[:120],
                'issues': [(i.get('word',''), i.get('rule',''), i.get('message','')[:60]) for i in issues[:3]]
            })

print("=" * 80)
print(f"FALSE NEGATIVES by category ({sum(len(v) for v in fn_categories.values())} total)")
print("=" * 80)
for cat in sorted(fn_categories.keys(), key=lambda x: -len(fn_categories[x])):
    items = fn_categories[cat]
    print(f"\n--- {cat.upper()} ({len(items)} FNs) ---")
    for t in items[:6]:
        print(f"  \"{t}\"")
    if len(items) > 6:
        print(f"  ... and {len(items)-6} more")

if fp_categories:
    print("\n" + "=" * 80)
    print(f"FALSE POSITIVES by category ({sum(len(v) for v in fp_categories.values())} total)")
    print("=" * 80)
    for cat in sorted(fp_categories.keys(), key=lambda x: -len(fp_categories[x])):
        items = fp_categories[cat]
        print(f"\n--- {cat.upper()} ({len(items)} FPs) ---")
        for item in items[:3]:
            print(f"  \"{item['text']}\"")
            for w, r, m in item['issues']:
                print(f"    -> [{r}] {m}")
