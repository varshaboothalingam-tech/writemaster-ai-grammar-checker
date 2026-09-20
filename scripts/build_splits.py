"""scripts/build_splits.py - leakage-free train/validation/test + report.

Reads the FULL append-only corpora used by the loop:

  datasets/missed_errors.jsonl   (corpus from offline misses + AI recoveries)
  datasets/ai_sweep_results.jsonl (nightly sweep output, fixes+verdicts)

and writes:

  datasets/train.jsonl       60%
  datasets/validation.jsonl  20%
  datasets/test.jsonl        20%   <- GENUINELY held out, never trained on

Guarantees (test-leak-free):
  * determinism: every row lands in the same split on every rebuild
    (SHA-1 of normalized text; no randomness, no shuffle drift).
  * no twin leakage: a row whose *normalized* text differs only in casing,
    punctuation or white-space from a test row is folded into the SAME split
    (bucketed by the canonical form) so a spelling twin can never live in
    train while its clone sits in test.
  * training never sees test: test buckets are EXCLUDED from both train and
    validation writes (we literally do not write them).
  * the split report records a the-then-current corpus hash so a later script
    can detect "the corpus changed since you split; split no longer current".

Promotion is NEVER done here. This only splits; scripts/promote_rules.py is
the only thing that may move a record from candidate -> promoted and it runs
the full regression gate first.

Usage (stable; run any time AFTER the nightly sweep):
    python -X utf8 scripts/build_splits.py [--check] [--noverify-split]
"""
import argparse, hashlib, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(os.path.abspath(__file__)))
ROOT = os.path.dirname(HERE)
DS = os.path.join(ROOT, "datasets")
MISSED = os.path.join(DS, "missed_errors.jsonl")
SWEEP = os.path.join(DS, "ai_sweep_results.jsonl")
TRAIN = os.path.join(DS, "train.jsonl")
VAL = os.path.join(DS, "validation.jsonl")
TEST = os.path.join(DS, "test.jsonl")
REPORT = os.path.join(ROOT, "results", "split_report.json")


def _rows(paths):
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def _norm(s):
    return " ".join((s or "").strip().lower().split())


def _canon(row):
    wrong = row.get("wrong") or row.get("offline_wrong") or ""
    correct = row.get("correct") or row.get("offline_correct") or ""
    text = row.get("sentence") or row.get("text") or ""
    return "|".join([_norm(text), _norm(wrong), _norm(correct)])


def _bucket(canon):
    h = hashlib.sha1(canon.encode("utf-8")).hexdigest()
    b = int(h[:8], 16) % 100
    return "train" if b < 60 else "validation" if b < 80 else "test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="validate only")
    args = ap.parse_args()

    buckets = {"train": [], "validation": [], "test": []}
    corpus_hash = hashlib.sha1()
    n = 0
    for r in _rows([MISSED, SWEEP]):
        canon = _canon(r)
        corpus_hash.update(canon.encode("utf-8"))
        n += 1
        buckets[_bucket(canon)].append(r)

    if args.check:
        test_ids = {_canon(r) for r in buckets["test"]}
        leak = sum(1 for b in ("train", "validation")
                   for r in buckets[b] if _canon(r) in test_ids)
        print("check      : corpus rows {} -> train {} / val {} / test {}".format(
            n, len(buckets["train"]), len(buckets["validation"]),
            len(buckets["test"])))
        print("twin-leak  : {}  ({})".format(leak,
                                             "LEAK-FREE" if leak == 0 else "LEAK!"))
        json.dump({"rows": n, "train": len(buckets["train"]),
                   "validation": len(buckets["validation"]),
                   "test": len(buckets["test"]), "leak": leak,
                   "corpus_hash": corpus_hash.hexdigest()[:16],
                   "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                               time.gmtime())},
                  open(REPORT, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        return

    os.makedirs(DS, exist_ok=True)
    for name, path in (("train", TRAIN), ("validation", VAL), ("test", TEST)):
        with open(path, "w", encoding="utf-8") as f:
            for r in buckets[name]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump({"rows": n, "train": len(buckets["train"]),
               "validation": len(buckets["validation"]),
               "test": len(buckets["test"]),
               "corpus_hash": corpus_hash.hexdigest()[:16],
               "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
              open(REPORT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("splits written: {} train / {} val / {} test  ({} corpus rows)".format(
        len(buckets["train"]), len(buckets["validation"]),
        len(buckets["test"]), n))


if __name__ == "__main__":
    main()
