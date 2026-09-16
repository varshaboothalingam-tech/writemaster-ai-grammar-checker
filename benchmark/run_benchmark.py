"""
run_benchmark.py — full honest benchmark over error_cases.json + clean_cases.json.

Usage:
    python benchmark/run_benchmark.py                       # train + val (default)
    python benchmark/run_benchmark.py --heldout             # held-out only
    python benchmark/run_benchmark.py --show 20             # show 20 FPs/FNs
    python benchmark/run_benchmark.py --ai                  # also run AI validator

Output: benchmark/benchmark_results.json + console summary.
"""
import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from new_pipeline import check_v4  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_FRAC, VAL_FRAC = 0.6, 0.2   # remainder is held-out


def _normalize(s: str) -> str:
    """Lowercase, collapse whitespace, strip sentence punctuation."""
    return re.sub(r"\s+", " ", s or "").strip().lower().strip(".,!?;:'\"()")


def apply_fixes(text: str, issues) -> str:
    """Apply returned replacements (end->start) to reproduce the corrected text."""
    out = text
    for iss in sorted(issues, key=lambda i: (i.get("start", 0) or 0), reverse=True):
        repl = iss.get("replacement")
        if not repl:
            continue
        s, e = iss.get("start", 0), iss.get("end", 0)
        if e <= s:
            continue
        out = out[:s] + repl + out[e:]
    return out


def load():
    with open(os.path.join(HERE, "error_cases.json"), encoding="utf-8") as f:
        errs = json.load(f)
    with open(os.path.join(HERE, "clean_cases.json"), encoding="utf-8") as f:
        cleans = json.load(f)
    out = []
    for e in errs:
        out.append({"id": e["id"], "text": e["text"], "should_flag": True,
                    "cat": e["note"], "good": e["good"]})
    for c in cleans:
        out.append({"id": c["id"], "text": c["text"], "should_flag": False,
                    "cat": "clean", "good": ""})
    return out


def split(cases):
    rng = random.Random(42)
    rng.shuffle(cases)
    n = len(cases)
    return (cases[:int(n * TRAIN_FRAC)],
            cases[int(n * TRAIN_FRAC):int(n * (TRAIN_FRAC + VAL_FRAC))],
            cases[int(n * (TRAIN_FRAC + VAL_FRAC)):])


def metrics(results):
    tp = sum(1 for r in results if r["result"] == "TP")
    fp = sum(1 for r in results if r["result"] == "FP")
    fn = sum(1 for r in results if r["result"] == "FN")
    tn = sum(1 for r in results if r["result"] == "TN")
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    fpr = fp / max(fp + tn, 1)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": p,
            "recall": r, "f1": f1, "fpr": fpr}


def run(cases, label, show=0, use_ai=False):
    results = []
    t0 = time.time()
    corr_ok = 0
    corr_total = 0
    ai_validated = 0
    ai_approved = 0
    ai_rejected = 0
    ai_cases = 0
    by_cat = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0,
                                  "corr_ok": 0, "corr_total": 0})
    for i, c in enumerate(cases):
        issues = []
        meta = {}
        try:
            resp = check_v4(c["text"], use_ai=use_ai)
            issues = resp["errors"]
            meta = resp.get("meta", {})
        except Exception as e:
            print(f"  !! case {c['id']} error: {e}")
        if meta.get("ai_used"):
            ai_cases += 1
            ai_validated += meta.get("ai_validated") or 0
            ai_approved += meta.get("ai_approved") or 0
            ai_rejected += meta.get("ai_rejected") or 0
        flagged = len(issues) > 0
        res = ("TP" if c["should_flag"] and flagged
               else "FN" if c["should_flag"] and not flagged
               else "FP" if not c["should_flag"] and flagged
               else "TN")
        bucket = by_cat[c["cat"]]
        bucket[res.lower()] += 1
        if res == "TP" and c.get("good"):
            fixed = apply_fixes(c["text"], issues)
            ok = _normalize(fixed) == _normalize(c["good"])
            bucket["corr_total"] += 1
            corr_total += 1
            if ok:
                bucket["corr_ok"] += 1
                corr_ok += 1
        results.append({"id": c["id"], "text": c["text"], "cat": c["cat"],
                        "should_flag": c["should_flag"], "result": res,
                        "num_issues": len(issues),
                        "good": c.get("good", ""),
                        "issues": [{"rule": e.get("rule_id"),
                                    "repl": e.get("replacement"),
                                    "cat": e.get("category"),
                                    "sources": e.get("sources"),
                                    "msg": e.get("message", "")[:70]}
                                   for e in issues[:5]]})
        if (i + 1) % 200 == 0:
            print(f"  {label}: {i+1}/{len(cases)} ({time.time()-t0:.0f}s)")
    m = metrics(results)
    m["correction_ok"] = corr_ok
    m["correction_total"] = corr_total
    m["correction_accuracy"] = (corr_ok / corr_total) if corr_total else 0.0
    m["by_category"] = {k: dict(v) for k, v in by_cat.items()}
    print(f"\n=== {label} ===")
    print(f"  TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")
    print(f"  Precision={m['precision']:.3f}  Recall={m['recall']:.3f}  "
          f"F1={m['f1']:.3f}  FPR={m['fpr']:.3f}")
    print(f"  Correction accuracy: {m['correction_ok']}/{m['correction_total']} "
          f"({m['correction_accuracy']:.3f})")
    if ai_validated:
        print(f"  Gemini/AI: validated={ai_validated} approved={ai_approved} "
              f"rejected={ai_rejected} "
              f"Approval Rate={ai_approved/ai_validated:.3f} "
              f"Rejection Rate={ai_rejected/ai_validated:.3f} "
              f"(cases with AI={ai_cases})")
    m["ai_validated"] = ai_validated
    m["ai_approved"] = ai_approved
    m["ai_rejected"] = ai_rejected

    if show:
        fps = [r for r in results if r["result"] == "FP"][:show]
        fns = [r for r in results if r["result"] == "FN"][:show]
        if fps:
            print(f"\n  --- FALSE POSITIVES ({len(fps)} shown) ---")
            for r in fps:
                print(f"  [{r['id']}] {r['text']}")
                for iss in r["issues"]:
                    print(f"     -> {iss['rule']}: {iss['msg']} src={iss['sources']} repl={iss['repl']}")
        if fns:
            print(f"\n  --- FALSE NEGATIVES ({len(fns)} shown) ---")
            for r in fns:
                print(f"  [{r['id']}] {r['text']}  (expected: {r['cat']})")
    return results, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", action="store_true")
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--ai", action="store_true",
                    help="run with the configured AI validator (Gemini/ollama/...); "
                         "requires the provider to be configured at runtime")
    args = ap.parse_args()

    cases = load()
    train, val, held = split(cases)
    print(f"train={len(train)} val={len(val)} heldout={len(held)}")

    if args.heldout:
        results, m = run(held, "HELDOUT", args.show, use_ai=args.ai)
    else:
        results, m = run(train, "TRAIN", args.show, use_ai=args.ai)
        vres, vm = run(val, "VAL", min(args.show, 5), use_ai=args.ai)
        # combine for a single reported number
        combined = results + vres
        cm = metrics(combined)
        cm["correction_ok"] = m["correction_ok"] + vm["correction_ok"]
        cm["correction_total"] = m["correction_total"] + vm["correction_total"]
        cm["correction_accuracy"] = (cm["correction_ok"] / cm["correction_total"]
                                     if cm["correction_total"] else 0.0)
        cm["ai_validated"] = m.get("ai_validated", 0) + vm.get("ai_validated", 0)
        cm["ai_approved"] = m.get("ai_approved", 0) + vm.get("ai_approved", 0)
        cm["ai_rejected"] = m.get("ai_rejected", 0) + vm.get("ai_rejected", 0)
        cm["by_category"] = {}
        for k in set(m["by_category"]) | set(vm["by_category"]):
            a = m["by_category"].get(k, {})
            b = vm["by_category"].get(k, {})
            cm["by_category"][k] = {
                "tp": a.get("tp", 0) + b.get("tp", 0),
                "fp": a.get("fp", 0) + b.get("fp", 0),
                "fn": a.get("fn", 0) + b.get("fn", 0),
                "tn": a.get("tn", 0) + b.get("tn", 0),
                "corr_ok": a.get("corr_ok", 0) + b.get("corr_ok", 0),
                "corr_total": a.get("corr_total", 0) + b.get("corr_total", 0),
            }
        print(f"\n--- COMBINED TRAIN+VAL ---")
        print(f"  TP={cm['tp']} FP={cm['fp']} FN={cm['fn']} TN={cm['tn']}  "
              f"P={cm['precision']:.3f} R={cm['recall']:.3f} F1={cm['f1']:.3f} FPR={cm['fpr']:.3f}")
        print(f"  Correction accuracy: {cm['correction_ok']}/{cm['correction_total']} "
              f"({cm['correction_accuracy']:.3f})")
        if cm.get("ai_validated"):
            av = cm["ai_validated"]
            print(f"  Gemini/AI: validated={av} approved={cm['ai_approved']} "
                  f"rejected={cm['ai_rejected']} "
                  f"Approval Rate={cm['ai_approved']/av:.3f} "
                  f"Rejection Rate={cm['ai_rejected']/av:.3f}")
        for cat in sorted(cm["by_category"]):
            s = cm["by_category"][cat]
            p = s["tp"] / max(s["tp"] + s["fp"], 1)
            r = s["tp"] / max(s["tp"] + s["fn"], 1)
            ca = (s["corr_ok"] / s["corr_total"]) if s["corr_total"] else 0.0
            print(f"    {cat:28s} TP={s['tp']:3d} FP={s['fp']} FN={s['fn']:3d} "
                  f"TN={s['tn']:3d}  P={p:.2f} R={r:.2f} corrAcc={ca:.2f}")

    report = {"generated": "2026-09-15",
              "train": train.__len__(), "val": val.__len__(),
              "heldout": held.__len__(), "results": results,
              "metrics": (m if args.heldout else cm)}
    out = os.path.join(HERE, "benchmark_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()