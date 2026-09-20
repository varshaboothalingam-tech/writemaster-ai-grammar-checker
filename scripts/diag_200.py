# scripts/diag_200.py - quick honest 200-error diagnostic (offline, fast)
import json, sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pipeline.master import check_master
errs = json.load(open("benchmark/error_cases.json", encoding="utf-8"))
random.seed(11)
samp = random.sample(errs, 200)
tp = fn = fp = exact = exact_n = 0
gold_mismatch = []   # flagged but fix wrong
missed = []          # should flag, didn't
for c in samp:
    r = check_master(c["text"], use_ai=False)
    fl = r.get("errors") or []
    flagged = len(fl) > 0
    sh = c.get("should_flag")
    if sh and flagged:
        tp += 1
        fixed = (r.get("corrected_text") or "").strip().lower()
        gold = (c.get("good") or "").strip().lower()
        if fixed == gold:
            exact += 1
        else:
            gold_mismatch.append({"id": c.get("id"), "in": c["text"][:60],
                                  "fixed": r["corrected_text"][:60],
                                  "gold": c["good"][:60]})
    elif sh and not flagged:
        fn += 1
        missed.append({"id": c.get("id"), "in": c["text"][:70]})
    else:
        fp += 1
recall = tp / max(tp + fn, 1)
exact_acc = exact / max(tp, 1)
print("=== DIAG 200 error cases (offline, no AI) ===")
print("recall {:.3f} ({}/{})   exact-fix {:.3f} ({}/{})   fp {}".format(
    recall, tp, tp + fn, exact_acc, exact, tp, fp))
print("mutant? tp+fn should be 200-should-less ->", tp + fn)
print("\n-- wrong-fix (flagged, gold differs) --")
for m in gold_mismatch[:8]:
    print("  id={} IN:'{}'  FIXED:'{}'  GOLD:'{}'".format(m["id"], m["in"], m["fixed"], m["gold"]))
print("\n-- missed (should flag, didn't) --")
for m in missed[:8]:
    print("  id={}  IN:'{}'".format(m["id"], m["in"]))
