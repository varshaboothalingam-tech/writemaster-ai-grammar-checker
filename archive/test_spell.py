from unified_pipeline import UnifiedPipeline
p = UnifiedPipeline()
import json

# Check if spelling dict loaded
print(f"Spelling map size: {len(p.spelling_map)}")
print(f"Sample entries: {dict(list(p.spelling_map.items())[:5])}")

# Check each word
test_words = ["mosee", "hannging", "correctcot", "whty", "prpnle"]
for w in test_words:
    found = w in p.spelling_map
    print(f"  '{w}' in dict: {found}")

# Now test the full pipeline
text = "hey check mosee laptop hannging correctcot check whty the prpnle"
results = p.check(text)
spelling_hits = [r for r in results if "SPELL" in r["rule_id"]]
print(f"\nSpelling errors found: {len(spelling_hits)}")
for r in results:
    print(f"  [{r['rule_id']}] {r['message']} -> {r.get('suggestion', 'N/A')}")
