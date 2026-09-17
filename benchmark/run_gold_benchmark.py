"""run_gold_benchmark.py — Phase 12–14: precision/recall/F1 on the real gold
datasets (not synthetic corpus).

Uses the kept input datasets:
    benchmark/error_cases.json  2000 flagged sentences, each with `good` target
    benchmark/clean_cases.json  2000 clean sentences (should NOT be flagged)

Measures, per item, at the SPAN level:
  - detected   = a proposal whose applied fix reproduces the gold `good` text
                 (exact span-correct repair)
  - repaired   = corrected_text == good (sentence-level perfect repair)
And on clean text: any error proposal => false positive sentence.

Output:
    results/gold_benchmark_<mode>.json    (full detail + metrics)
    console table                          (P / R / F1 / FP / repair).

Usage:
    python benchmark/run_gold_benchmark.py            # offline rule path
    python benchmark/run_gold_benchmark.py --mode ai  # live Gemini (needs key)
    python benchmark/run_gold_benchmark.py --limit 200
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("AI_PROVIDER", "none")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pipeline.master import check_master  # noqa: E402


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def span_fix_reproduces_good(text: str, good: str, errors: List[Dict]) -> bool:
    """True if a single error proposal, applied at its span, reproduces `good`."""
    for e in errors or []:
        w = str(e.get("wrong") or "")
        c = str(e.get("correct") or "")
        if not w or not c:
            continue
        s, end = e.get("start"), e.get("end")
        try:
            rebuilt = text[: s] + c + text[end:]
        except TypeError:
            rebuilt = text.replace(w, c, 1)
        if norm(rebuilt) == norm(good):
            return True
    return False


def evaluate(items: List[Dict], clean: List[Dict], mode: str) -> Dict:
    tp = fp_total = fn = 0
    repaired = clean_fp = total_proposals = 0
    elapsed = time.time()
    per_item: List[Dict] = []

    for it in items:
        text = it["text"]
        good = it.get("good") or ""
        res = check_master(text, use_ai=mode == "ai")
        errors = res.get("errors") or []
        total_proposals += len(errors)
        detected = span_fix_reproduces_good(text, good, errors)
        is_repaired = bool(good and norm(res.get("corrected_text", "")) == norm(good))
        repaired += int(is_repaired)
        if detected:
            tp += 1
            fp_total += max(0, len(errors) - 1)
        else:
            fn += 1
            fp_total += len(errors)
        per_item.append({
            "id": it.get("id"),
            "text": text,
            "good": good,
            "note": it.get("note", ""),
            "detected": detected,
            "repaired": is_repaired,
            "corrected": res.get("corrected_text", ""),
            "consensus": res.get("consensus", {}),
            "quality": res.get("quality", {}),
            "proposals": [{
                "wrong": e.get("wrong"), "correct": e.get("correct"),
                "type": e.get("type"), "category": e.get("category"),
                "confidence": e.get("confidence"), "consensus": e.get("consensus"),
            } for e in errors],
        })

    for it in clean:
        res = check_master(it["text"], use_ai=mode == "ai")
        n = len(res.get("errors") or [])
        if n:
            clean_fp += 1
        fp_total += n
        total_proposals += n

    n_err = len(items)
    precision = tp / total_proposals if total_proposals else 0.0
    recall = tp / n_err if n_err else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "config": {"mode": mode, "error_items": n_err, "clean_items": len(clean)},
        "metrics": {
            "true_positives": tp,
            "false_negatives": fn,
            "false_positives": fp_total,
            "total_proposals": total_proposals,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "sentence_repair_accuracy": round(repaired / n_err, 4) if n_err else 0.0,
            "clean_fp_sentences": clean_fp,
            "clean_fp_rate": round(clean_fp / len(clean), 4) if clean else 0.0,
            "elapsed_seconds": round(time.time() - elapsed, 2),
        },
        "errors": per_item,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["offline", "ai"], default="offline")
    ap.add_argument("--limit", type=int, default=0, help="0 = full datasets")
    args = ap.parse_args()

    with open(os.path.join(ROOT, "benchmark", "error_cases.json"), encoding="utf-8") as f:
        errors = json.load(f)
    with open(os.path.join(ROOT, "benchmark", "clean_cases.json"), encoding="utf-8") as f:
        clean = json.load(f)
    if args.limit:
        errors = errors[: args.limit]
        clean = clean[: args.limit]

    print(f"gold benchmark mode={args.mode}: {len(errors)} error + {len(clean)} clean")
    results = evaluate(errors, clean, args.mode)

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    out = os.path.join(ROOT, "results", f"gold_benchmark_{args.mode}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    m = results["metrics"]
    print("\n=== Gold-set metrics ===")
    print(f"TP = {m['true_positives']}   FN = {m['false_negatives']}   "
          f"FP (proposal-level) = {m['false_positives']}   proposals = {m['total_proposals']}")
    print(f"Precision = {m['precision']:.2%}   Recall = {m['recall']:.2%}   F1 = {m['f1']:.2%}")
    print(f"Sentence repair = {m['sentence_repair_accuracy']:.2%}   "
          f"clean FP rate = {m['clean_fp_rate']:.2%} ({m['clean_fp_sentences']}/{len(clean)})")
    print(f"elapsed = {m['elapsed_seconds']}s   wrote: {out}")

    under = [e for e in results["errors"] if not e["detected"]][:6]
    if under:
        print("\nSample false negatives:")
        for e in under:
            print("  -", (e["text"][:60]), "|| good:", (e["good"] or "")[:40],
                  "|| got:", (e["corrected"] or "")[:40])


if __name__ == "__main__":
    main()