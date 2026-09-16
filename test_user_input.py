from unified_pipeline import UnifiedPipeline
p = UnifiedPipeline()
text = "hey check mosee laptop hannging correctcot check whty the prpnle"
results = p.check(text)
print(f"Input: {text}")
print(f"Errors found: {len(results)}")
for r in results:
    rule = r["rule_id"]
    msg = r["message"]
    sugg = r.get("suggestion", "N/A")
    print(f"  [{rule}] {msg}")
    print(f"    Suggestion: {sugg}")
