import json, urllib.request

text = ("Yesterday my friend and me goes to a large stopping mall because we waited "
        "to buy some new clothe. When we reach the mall, there was many people writing "
        "outside because the stops was not spend yet.")

req = urllib.request.Request(
    "http://127.0.0.1:5001/api/check",
    data=json.dumps({"text": text}).encode(),
    headers={"Content-Type": "application/json"},
)
resp = urllib.request.urlopen(req)
data = json.loads(resp.read())

issues = data["issues"]
print("REGRESSION TEST - Total issues:", len(issues))
for e in sorted(issues, key=lambda x: x.get("start_position", 0)):
    print("  %s | %3d-%3d | %-20s -> %-20s | conf=%.2f | %s" % (
        e["type"], e.get("start_position", 0), e.get("end_position", 0),
        e["word"], e["corrected_text"], e["confidence"], e["rule"]))

print("\n--- CORRECT CONTENT TEST ---")
text2 = ("Yesterday my friend and I went to the shopping mall. There were many people "
         "waiting outside because the shops were not open yet.")
req2 = urllib.request.Request(
    "http://127.0.0.1:5001/api/check",
    data=json.dumps({"text": text2}).encode(),
    headers={"Content-Type": "application/json"},
)
resp2 = urllib.request.urlopen(req2)
data2 = json.loads(resp2.read())
issues2 = data2["issues"]
print("Total issues:", len(issues2))
if issues2:
    for e in issues2:
        print("  %s | %s -> %s | %s" % (e["type"], e["word"], e["corrected_text"], e["rule"]))

print("\n--- WRITING NOT CHANGED TEST ---")
text3 = "The students were writing outside while they waited for the teacher."
req3 = urllib.request.Request(
    "http://127.0.0.1:5001/api/check",
    data=json.dumps({"text": text3}).encode(),
    headers={"Content-Type": "application/json"},
)
resp3 = urllib.request.urlopen(req3)
data3 = json.loads(resp3.read())
issues3 = data3["issues"]
writing_issues = [e for e in issues3 if e["word"].lower() == "writing"]
print("Total issues:", len(issues3))
if writing_issues:
    print("FAIL: 'writing' was incorrectly flagged!")
    for e in writing_issues:
        print("  %s | %s -> %s | %s" % (e["rule"], e["word"], e["corrected_text"], e["message"]))
else:
    print("PASS: 'writing' was NOT incorrectly flagged")
