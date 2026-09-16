import time, requests, json

text = "my laptop is hannging mouse working slow"

# Step 1: Check
t1 = time.time()
r1 = requests.post("http://127.0.0.1:5001/api/check-v3", json={"text": text})
d1 = r1.json()
print(f"CHECK: {time.time()-t1:.2f}s  issues={len(d1.get('issues',[]))}")
for iss in d1.get('issues', []):
    print(f"  [{iss['rule_id']}] word='{iss.get('word','')}' start={iss.get('start_position',iss.get('start',''))} end={iss.get('end_position',iss.get('end',''))} replacement='{iss.get('replacement','')}' conf={iss.get('confidence','')}")

# Step 2: Fix all
t2 = time.time()
r2 = requests.post("http://127.0.0.1:5001/api/fix-all", json={"text": text})
d2 = r2.json()
print(f"\nFIX: {time.time()-t2:.2f}s  corrected='{d2.get('corrected','')}'")
print(f"  changes: {d2.get('changes',[])}")

# Step 3: Re-check the corrected text
corrected = d2.get('corrected', '')
if corrected:
    t3 = time.time()
    r3 = requests.post("http://127.0.0.1:5001/api/check-v3", json={"text": corrected})
    d3 = r3.json()
    print(f"\nRE-CHECK: {time.time()-t3:.2f}s  issues={len(d3.get('issues',[]))}")
    for iss in d3.get('issues', []):
        print(f"  [{iss['rule_id']}] word='{iss.get('word','')}' replacement='{iss.get('replacement','')}' conf={iss.get('confidence','')}")
