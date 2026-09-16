import json
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import generate_benchmark as gb
from new_pipeline import check_v4


def apply_fixes(text, issues):
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


def norm(s):
    return re.sub(r"\s+", " ", s or "").strip().lower().strip(".,!?;:'\"()")


NEW_ERR = ["modal-form", "collective-have-sva", "tense-backshift",
           "scheduled-on", "quantifier-relative-sva", "duplicate-verb",
           "tense-marker-perfect", "passive-sva"]
NEW_CLEAN = ["modals correct", "collective-has + BrE tolerance", "backshift clean / no past marker",
             "scheduled on calendar/system / for", "anybody/anyone + plural who-were clean",
             "duplicate verb clean", "present perfect correct",
             "present perfect + already correct", "passive agreement correct"]


def main():
    err_counts = {n: 0 for n, _ in gb.ERROR_CATS}
    for name, fn in gb.ERROR_CATS:
        err_counts[name] = len(list(fn()))

    print("=== ERROR CATEGORY YIELDS ===")
    for name, fn in gb.ERROR_CATS:
        mark = "<-NEW" if name in NEW_ERR else ""
        print(f"  {name:35s} {err_counts[name]:4d} {mark}")

    clean_counts = {n: 0 for n, _ in gb.CLEAN_GENS}
    for note, fn in gb.CLEAN_GENS:
        clean_counts[note] = len(list(fn()))
    print("\n=== CLEAN GEN YIELDS ===")
    for note, fn in gb.CLEAN_GENS:
        mark = "<-NEW" if note in NEW_CLEAN else ""
        print(f"  {note:45s} {clean_counts[note]:4d} {mark}")

    print("\n=== NEW ERROR CHECK (flagged + correct fix) ===")
    total_bad = 0
    bad_fails = []
    for name in NEW_ERR:
        fn = dict(gb.ERROR_CATS)[name]
        this_cat_bad = 0
        this_ok = 0
        for bad, good, note in fn():
            total_bad += 1
            this_cat_bad += 1
            errs = check_v4(bad)["errors"]
            if not errs:
                bad_fails.append((name, bad, "NO FLAG"))
                continue
            fixed = apply_fixes(bad, errs)
            if norm(fixed) != norm(good):
                bad_fails.append((name, bad, f"FIX {fixed!r} != {good!r}"))
            else:
                this_ok += 1
        print(f"  {name:35s} ok={this_ok}/{this_cat_bad}")

    print("\n=== NEW CLEAN CHECK (must stay unflagged) ===")
    total_clean = 0
    clean_fails = []
    for note in NEW_CLEAN:
        fn = dict(gb.CLEAN_GENS)[note]
        this_n = 0
        this_ok = 0
        for s in fn():
            total_clean += 1
            this_n += 1
            errs = check_v4(s)["errors"]
            if errs:
                clean_fails.append((note, s, [e["rule_id"] for e in errs]))
            else:
                this_ok += 1
        print(f"  {note:45s} ok={this_ok}/{this_n}")

    print(f"\nTOTAL new error cases  : {total_bad}  failures: {len(bad_fails)}")
    for name, bad, why in bad_fails[:25]:
        print(f"    [{name}] {bad!r} -> {why}")
    print(f"TOTAL new clean cases  : {total_clean}  failures: {len(clean_fails)}")
    for note, s, rules in clean_fails[:25]:
        print(f"    [{note}] {s!r} -> {rules}")


if __name__ == "__main__":
    main()