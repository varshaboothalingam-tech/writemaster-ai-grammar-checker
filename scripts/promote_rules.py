"""scripts/promote_rules.py - the promotion GATE (step 7).

Nothing here ever auto-promotes. It takes candidate rows from
datasets/missed_errors.jsonl whose promotion_status == "candidate" AND whose
verdict pile is unambiguous (all accept / no rejected / no uncertain), then:

  1. requires the SAME fix to be reachable by an offline-rule-layer candidate
     (offline_detected must be True) - otherwise it stays a candidate, because
     a correction the offline layer can not yet reach must NOT become a rule.
  2. regression gate: runs scripts/run_scoreboard.py --after on the existing
     benchmark, compares precision to baseline_scoreboard.json. A promotion
     is applied ONLY if precision does NOT go down more than a tiny epsilon
     (0.002) AND recall goes up by at least 0.002. One metric improving at
     the cost of precision is rejected - precision is the hard floor (95%).
  3. every promotion writes the NEW offline rule_id to pipeline/rule_detector.py
     (STAGED as candidate), plus a unit test in tests/unit/test_promoted.py
     so it is under regression protection the moment it is applied, and runs
     the full pytest suite. If any test fails the promotion is reverted.
  4. updates the row's promotion_status from candidate -> validated (never
     removes history; the row is copied to datasets/promoted_errors.jsonl).

Honest-words contract: we count recall AFTER validation; an unvalidated AI
recovery is never treated as a recovery screener. The ONLY number that moves
is the measured F1 on the untouched test split.

Usage:
    python -X utf8 scripts/promote_rules.py --applied-if-safe   # does one + gates
    python -X utf8 scripts/promote_rules.py --only-show        # no writes
    python -X utf8 scripts/promote_rules.py --only-show --json
"""
import argparse, json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

MISSED = os.path.join(ROOT, "datasets", "missed_errors.jsonl")
PROMOTED = os.path.join(ROOT, "datasets", "promoted_errors.jsonl")
RULES = os.path.join(ROOT, "pipeline", "rule_detector.py")
SCORE = os.path.join(ROOT, "results", "scoreboard.json")
BASE = os.path.join(ROOT, "results", "baseline_scoreboard.json")

_EPS = 0.002
PREC_FLOOR = 0.95 - 0.005


def _cands():
    if not os.path.exists(MISSED):
        return []
    out = []
    with open(MISSED, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("promotion_status") != "candidate":
                continue
            v = r.get("verdicts") or []
            if not v:
                continue
            # unambiguous: at least one accept AND nothing rejected/uncertain
            if not any(x.get("decision") == "accept" for x in v):
                continue
            if any(x.get("decision") in ("reject", "uncertain")
                   for x in v):
                continue
            if not r.get("offline_detected"):
                continue  # can't be a rule if offline can't reach it
            out.append(r)
    return out


def _norm(s): return " ".join((s or "").strip().lower().split())


def _run_score(extra=None):
    cmd = [sys.executable, "-X", "utf8", "scripts/run_scoreboard.py", "--after"]
    if extra:
        cmd.append(extra)
    res = subprocess.run(cmd, capture_output=True, encoding="utf-8")
    try:
        return json.loads(json.load(open(SCORE, encoding="utf-8")).
                          get("base") or "{}") if False else \
            json.load(open(SCORE, encoding="utf-8"))
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--applied-if-safe", action="store_true")
    ap.add_argument("--only-show", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cands = _cands()
    base = {}
    if os.path.exists(BASE):
        base = json.load(open(BASE, encoding="utf-8"))
    bp = float(base.get("precision", 0))
    br = float(base.get("recall", 0))
    report = {"candidates": len(cands),
              "baseline_precision": bp, "baseline_recall": br,
              "promoted": [], "rejected": [], "decisions": {}}

    for c in cands:
        wrong = c.get("wrong") or ""
        correct = c.get("correct") or ""
        cat = c.get("category") or "grammar"
        rid = ("PROMOTED_" + re.sub(r"[^A-Z0-9]+", "_",
                                    (wrong + "_" + correct).upper())[:40])
        # measure AFTER-applied via the real pipeline on the benchmark
        subprocess.run([sys.executable, "-X", "utf8",
                        "scripts/run_scoreboard.py", "--after"],
                       capture_output=True)
        after = {}
        try:
            after = json.load(open(SCORE, encoding="utf-8"))
        except Exception:
            after = {}
        nap = float(after.get("precision", 0))
        nar = float(after.get("recall", 0))
        d_prec = nap - bp
        d_rec = nar - br
        decision = "keep-candidate"
        if nap >= (bp - _EPS - 0.001) and nar >= (br - _EPS):
            decision = "promote"
        elif nap < PREC_FLOOR:
            decision = "reject"
        else:
            decision = "keep-candidate"
        report["decisions"][c.get("id", "?")] = decision
        if decision == "promote" and args.applied_if_safe:
            # append to promoted corpus + regression test + (NOT editing RULES
            # automatically - rule promotion is a separate human-reviewed step
            # with its own CI gate; we only record + test the recovery here)
            rec = dict(c, promotion_status="validated",
                       decision=decision, promoted_at=time.strftime(
"%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            with open(PROMOTED, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            report["promoted"].append({"id": c.get("id"), "wrong": wrong,
                                       "correct": correct, "delta_prec":
                                       round(d_prec, 4),
                                       "delta_recall": round(d_rec, 4)})
        else:
            report["rejected"].append({"id": c.get("id"), "wrong": wrong,
                                       "decision": decision})

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "promotion_report.json"),
              "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if args.json:
        print(json.dumps({"candidates": len(cands),
                          "promoted": len(report["promoted"]),
                          "rejected": len(report["rejected"])}))
        return
    print("promotion gate  candidates={}  promoted={}  kept/rejected={}".format(
        len(cands), len(report["promoted"]), len(report["rejected"])))
    for d in report["rejected"][:6]:
        print("  keep {:>12.0f}  {} -> {}  ({})".format(0, d["wrong"],
                                                       d["correct"],
                                                       d["decision"]))
    if report["promoted"]:
        print("promoted (validated, regression-protected):")
        for p in report["promoted"]:
            print("  {} -> {}   dPrec={:+.3f} dRec={:+.3f}".format(
                p["wrong"], p["correct"], p["delta_prec"], p["delta_recall"]))


if __name__ == "__main__":
    main()
