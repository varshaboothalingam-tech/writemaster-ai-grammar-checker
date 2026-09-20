"""scripts/run_scoreboard.py - deterministic BEFORE/AFTER scoreboard over the
full curated benchmark (benchmark/error_cases.json + clean_cases.json).

EXACT copy of the audit measurement, reading real error records
(keys wrong/correct/consensus/rule_id — see pipeline/schema.py).

Scores the OFFLINE engine only (no AI) so the result is stable in CI and
comparable over time. AI pass is layered on top in production and reported
separately by scripts/ai_sweep.py; this file deliberately measures what the
offline core alone does to keep before/after honest.

    python -X utf8 scripts/run_scoreboard.py --before --json   # baseline
    python -X utf8 scripts/run_scoreboard.py --after  --json   # current
    python -X utf8 scripts/run_scoreboard.py                   # verbose

Writes results/scoreboard.json + results/baseline_scoreboard.json
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.master import check_master  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BENCH = os.path.join(ROOT, "benchmark")
ERR = os.path.join(BENCH, "error_cases.json")
CLEAN = os.path.join(BENCH, "clean_cases.json")
RES = os.path.join(ROOT, "results")
SCORE = os.path.join(RES, "scoreboard.json")
BASELINE = os.path.join(RES, "baseline_scoreboard.json")


def _norm(s):
    return " ".join((s or "").strip().lower().split())


def _measure(cases):
    tp = fp = fn = tn = 0
    exact_ok = exact_n = 0
    rollup = {"ai_only": 0, "validated": 0, "uncertain": 0, "rejected": 0}
    for c in cases:
        should = bool(c.get("should_flag"))
        text = c.get("text", "")
        try:
            r = check_master(text, use_ai=False)
        except Exception:
            r = {"errors": []}
        errs = r.get("errors") or []
        flagged = bool(errs)
        if should and flagged:
            tp += 1
        elif should:
            fn += 1
        elif flagged:
            fp += 1
        else:
            tn += 1
        for e in errs:
            cons = (e.get("consensus") or "").upper()
            if cons == "AI_ONLY":
                rollup["ai_only"] += 1
            elif "VALIDAT" in cons:
                rollup["validated"] += 1
            elif "UNCERTAIN" in cons or "UNCERT" in cons:
                rollup["uncertain"] += 1
            elif "REJECT" in cons:
                rollup["rejected"] += 1
        good = c.get("good") or ""
        if should and good and flagged:
            exact_n += 1
            fixed = (r.get("corrected_text") or "").strip().lower()
            if _norm(fixed) == _norm(good):
                exact_ok += 1
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return {"precision": round(p, 4), "recall": round(r, 4),
            "f1": round(f1, 4),
            "exact_correction_accuracy": round(exact_ok / max(exact_n, 1), 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "rollup": rollup,
            "total": tp + fp + fn + tn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", action="store_true")
    ap.add_argument("--after", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    err = json.load(open(ERR, encoding="utf-8"))
    clean = json.load(open(CLEAN, encoding="utf-8"))
    score = _measure(err + clean)
    os.makedirs(RES, exist_ok=True)
    if args.before:
        json.dump(score, open(BASELINE, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        echo_mode = "baseline"
    else:
        json.dump({"baseline": _load(BASELINE), "current": score},
                  open(SCORE, "w", encoding="utf-8"), ensure_ascii=False,
                  indent=2)
        echo_mode = "after"
    if args.json:
        print(json.dumps({"precision": score["precision"],
                          "recall": score["recall"], "f1": score["f1"],
                          "exact_correction_accuracy":
                              score["exact_correction_accuracy"],
                          "tp": score["tp"], "fp": score["fp"],
                          "fn": score["fn"], "tn": score["tn"],
                          "rollup": score["rollup"], "mode": echo_mode}))
        return
    print("=== {0} scoreboard ({1} cases) ===".format(echo_mode, score["total"]))
    for k in ("precision", "recall", "f1", "exact_correction_accuracy"):
        print("  {0:<28} {1:.4f}".format(k, score[k]))
    print("  TP/FP/FN/TN              {0}/{1}/{2}/{3}".format(
        score["tp"], score["fp"], score["fn"], score["tn"]))
    print("  rollup AI_ONLY/VAL/UNC/REJ {0}/{1}/{2}/{3}".format(
        score["rollup"]["ai_only"], score["rollup"]["validated"],
        score["rollup"]["uncertain"], score["rollup"]["rejected"]))


def _load(path):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return {}


if __name__ == "__main__":
    main()
