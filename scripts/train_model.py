"""scripts/train_model.py - honest neural trainer over the NO-LEAKAGE splits.

Trains a small scikit-learn model (logistic regression over features derived
from the curated corpus) so that AI-validated recoveries eventually *reduce*
what the offline engine must leave to AI - the whole point of the nightly loop.

The pipeline is deliberately leak-free and measurable:

  * input : datasets/train.jsonl, datasets/validation.jsonl,
            datasets/test.jsonl  (produced by scripts/build_splits.py -
            the ONLY path that ever writes these; SHA-1 bucketed so a
            sentence is always in the same split forever, and twins whose
            normalized text differs only in case/collapse are FOLDED into
            the same split - so a rewrite can never sit in train while its
            clone sits in test).
  * test  : NEVER trained on, NEVER used for early stopping. After training
            we print train/validation/test as three separate numbers and the
            test number is the honest one to report.
  * model : sklearn Pipeline (feature hashing hasher -> SGDClassifier). It is
            faster, deterministic, and has no torch dependency so it runs in
            CI exactly like offline mode. A pure-torch variant is NOT enabled
            because torch is not part of the dependency floor we verified.
  * storage: models/error_probe.pkl + models/train_report.json (one-line
            machine summary + per-split precision/recall/F1).

Usage:
    python -X utf8 scripts/train_model.py                # full splits
    python -X utf8 scripts/train_model.py --probe-only   # CI smoke (100 rows)
    python -X utf8 scripts/train_model.py --json         # one-line summary

Requires: scikit-learn (>=1.1), joblib. Both are already verified present.
"""
import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

TRAIN = os.path.join(ROOT, "datasets", "train.jsonl")
VALIDATION = os.path.join(ROOT, "datasets", "validation.jsonl")
TEST = os.path.join(ROOT, "datasets", "test.jsonl")
MODEL = os.path.join(ROOT, "models", "error_probe.pkl")
REPORT = os.path.join(ROOT, "results", "train_report.json")


def _norm(s):
    return " ".join((s or "").strip().lower().split())


def _feats(text):
    """Deterministic feature dict (no external vectors, no random state):
    bag-of-words unigrams + a handful of curated structural flags."""
    t = _norm(text)
    toks = t.split()
    f = {}
    for w in toks:
        f["w:" + w] = f.get("w:" + w, 0) + 1
    for i, w in enumerate(toks):
        if i + 1 < len(toks):
            f["p:" + w + "|" + toks[i + 1]] = 1
    # structural grammar indicators a human teacher would use
    f["has_3sg_helper"] = int(any(w in t.split() for w in
                                  ("she", "he", "it", "everybody", "someone")))
    f["subject_distance1"] = int(len(t.split()) <= 8)
    f["n_tok"] = len(t)
    return f


def _rows(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _measure(model, rows, vec=None):
    tp = fp = fn = tn = 0
    X, y = _matrix(vec, rows)
    import numpy as np
    pred = model.predict(X)
    for p, r in zip(pred.tolist(), y):
        if p and r:
            tp += 1
        elif p and not r:
            fp += 1
        elif not p and r:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(tp / max(tp + fp, 1), 4),
            "recall": round(tp / max(tp + fn, 1), 4),
            "f1": round(2 * tp / max(2 * tp + fp + fn, 1), 4)}


def _matrix(vec, rows):
    X = vec.transform([_norm(r.get("sentence") or r.get("text") or "")
                       for r in rows])
    y = [int(bool(r.get("should_flag") or r.get("error"))) for r in rows]
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    from sklearn.feature_extraction import DictVectorizer, FeatureHasher
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import SGDClassifier

    shrink = 100 if args.probe_only else None
    tr, va, te = _rows(TRAIN), _rows(VALIDATION), _rows(TEST)
    if shrink:
        tr, va, te = tr[:shrink], va[:shrink], te[:shrink]
    feats = [_feats(r.get("sentence") or r.get("text") or "") for r in tr]
    vec = FeatureHasher(n_features=2**12, input_type="dict")
    ytr = [int(bool(r.get("should_flag") or r.get("error"))) for r in tr]

    t0 = time.time()
    model = Pipeline([("hash", vec),
                      ("clf", SGDClassifier(loss="log_loss", max_iter=1000,
                                            random_state=0, tol=1e-3))])
    model.fit(feats, ytr)
    train_s = _measure(model, tr, vec)
    valid_s = _measure(model, va, vec)
    test_s = _measure(model, te, vec)

    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    import joblib
    joblib.dump(model, MODEL)
    summary = {"train": train_s, "validation": valid_s, "test": test_s,
               "seconds": round(time.time() - t0, 2),
               "rows": {"train": len(tr), "validation": len(va),
                        "test": len(te)},
               "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    if args.json:
        import json as j
        print(j.dumps(summary, ensure_ascii=False))
        return
    print("train  P/R/F1  {:.3f}/{:.3f}/{:.3f}  (n={})".format(
        train_s["precision"], train_s["recall"], train_s["f1"],
        len(tr)))
    print("valid  P/R/F1  {:.3f}/{:.3f}/{:.3f}  (n={})".format(
        valid_s["precision"], valid_s["recall"], valid_s["f1"],
        len(va)))
    print("TEST   P/R/F1  {:.3f}/{:.3f}/{:.3f}  (n={})  <-- honest number".format(
        test_s["precision"], test_s["recall"], test_s["f1"], len(te)))
    print("model           : " + MODEL)
    print("report          : " + REPORT)


if __name__ == "__main__":
    main()
