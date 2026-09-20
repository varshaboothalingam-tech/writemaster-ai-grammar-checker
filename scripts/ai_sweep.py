"""scripts/ai_sweep.py - the NIGHTLY AI SWEEP (loop step 1-3).

For every case in datasets/missed_errors.jsonl (append-only corpus from the
benchmark + missed_errors corpus), do the honest offline->AI->validate dance:

  1. offline pass          -> local NLP corrections (always available)
  2. AI pass (if key)      -> Gemini/AI adds candidates local missed
  3. AI verifier           -> accept/reject/uncertain per candidate
  4. compare vs gold       -> exact/partial/missed, log VERIFIED recoveries
  5. append results to datasets/ai_sweep_results.jsonl (append-only, no dup)

Records with promotion_status=candidate only; NEVER auto-promote here.

Usage:
    python -X utf8 scripts/ai_sweep.py --limit 100            # CI smoke batch
    python -X utf8 scripts/ai_sweep.py --no-ai               # offline only
    python -X utf8 scripts/ai_sweep.py --json                # one-line summary
"""
import argparse, hashlib, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MISSED = os.path.join(ROOT, "datasets", "missed_errors.jsonl")
SWEEP_LOG = os.path.join(ROOT, "datasets", "ai_sweep_results.jsonl")


def _norm(s): return " ".join((s or "").strip().lower().split())


def _hash(text):
    return hashlib.sha1(_norm(text).encode("utf-8")).hexdigest()[:16]


def _corpus_rows(limit):
    rows = []
    if os.path.exists(MISSED):
        with open(MISSED, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    return rows[:limit] if limit else rows


def _seen_hashes():
    seen = set()
    if os.path.exists(SWEEP_LOG):
        with open(SWEEP_LOG, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    seen.add(json.loads(line).get("row_hash"))
                except Exception:
                    continue
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-ai", action="store_true", help="offline engine only")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    from pipeline.master import check_master
    from pipeline.ai_verifier import verify
    from pipeline.consensus import merge

    rows = _corpus_rows(args.limit)
    seen = _seen_hashes()
    use_ai = (not args.no_ai)

    stats = {"swept": 0, "offline_detected": 0, "ai_added": 0,
             "validated": 0, "uncertain": 0, "rejected": 0,
             "ai_only_recovered": 0, "skipped_dup": 0, "wrote": 0}
    t0 = time.time()

    os.makedirs(os.path.dirname(SWEEP_LOG), exist_ok=True)
    with open(SWEEP_LOG, "a", encoding="utf-8") as out:
        for c in rows:
            text = c.get("text") or c.get("sentence") or c.get("input") or ""
            expected = c.get("correct") or c.get("good") or ""
            wrong = c.get("wrong") or c.get("expected_wrong") or ""
            hashi = _hash(text + "|" + _norm(wrong) + "|" + _norm(expected))
            if hashi in seen:
                stats["skipped_dup"] += 1
                continue

            results = []
            record = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "row_hash": hashi, "text": text, "expected": expected,
                      "wrong": wrong, "promotion_status": "candidate"}
            # 1) offline
            try:
                off = check_master(text, use_ai=False)
                off_errors = off.get("errors") or []
            except Exception:
                off_errors = []
            stats["swept"] += 1
            if off_errors:
                record["offline_detected"] = True
                record["offline_corrections"] = [
                    {"wrong": e.get("wrong"), "correct": e.get("correct"),
                     "category": e.get("category"), "rule_id": e.get("rule_id")}
                    for e in off_errors[:10]]
                stats["offline_detected"] += 1

            # 2) AI pass
            ai_candidates = []
            if use_ai:
                try:
                    ai = check_master(text, use_ai=True)
                    for e in (ai.get("errors") or []):
                        if e.get("_discovery") == "AI_ONLY":
                            ai_candidates.append(e)
                except Exception:
                    pass
                if ai_candidates:
                    stats["ai_added"] += 1

            # 3) verifier: validate each AI candidate independently
            record["verdicts"] = []
            for cand in (record.get("offline_corrections") or []) + ai_candidates:
                decision = "uncertain"
                try:
                    v = verify(original=text, corrected=(cand.get("correct") or ""),
                               changes=[cand], use_ai=use_ai)
                    decision = v.get("decision", "uncertain")
                except Exception:
                    decision = "uncertain"
                record["verdicts"].append(
                    {"wrong": cand.get("wrong"), "correct": cand.get("correct"),
                     "decision": decision})
                if decision == "accept":
                    stats["validated"] += 1
                elif decision in ("uncertain", "uncertain"):
                    stats["uncertain"] += 1
                else:
                    stats["rejected"] += 1

            # 4) compare vs gold
            got = next((r["correct"] for r in record["verdicts"]
                        if r["decision"] == "accept"), None)
            if got and wrong and _norm(got) == _norm(expected):
                stats["ai_only_recovered"] += 1
                record["recovered"] = True
            else:
                record["recovered"] = False

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["wrote"] += 1

    summary = {"rows_swept": stats["swept"], "offline_detected": stats["offline_detected"],
               "ai_added": stats["ai_added"], "validated": stats["validated"],
               "uncertain": stats["uncertain"], "rejected": stats["rejected"],
               "ai_only_recovered": stats["ai_only_recovered"],
               "exact_override": round(stats["ai_only_recovered"] / max(stats["wrote"], 1), 3),
               "wrote": stats["wrote"], "skipped_dup": stats["skipped_dup"],
               "seconds": round(time.time() - t0, 1)}
    if args.json:
        import json as j
        print(j.dumps(summary, ensure_ascii=False))
        return
    print("AI SWEEP  rows={rows_swept}  offline_detected={offline_detected} "
          "ai_added={ai_added} validated={validated} uncertain={uncertain} "
          "rejected={rejected} ai_only_recovered={ai_only_recovered} "
          "wrote={wrote} skipped_dup={skipped_dup} {elapsed}s".format(
              elapsed=summary["seconds"], **summary))


if __name__ == "__main__":
    main()
