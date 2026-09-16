"""Evaluate grammar checker against the 1000-sentence CSV dataset."""
import csv
import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unified_pipeline import UnifiedPipeline

def load_dataset(path):
    sentences = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('mistake_text') and row.get('correct_text'):
                sentences.append({
                    'id': int(row['id']),
                    'category': row.get('category', ''),
                    'context': row.get('context', ''),
                    'mistake_text': row['mistake_text'].strip(),
                    'correct_text': row['correct_text'].strip(),
                })
    return sentences

def evaluate(dataset_path='grammar_error_dataset_1000.csv'):
    dataset = load_dataset(dataset_path)
    pipeline = UnifiedPipeline()

    detected = 0
    missed_ids = []
    missed_by_cat = {}
    details = []
    start = time.time()

    for item in dataset:
        result = pipeline.check(item['mistake_text'])
        rule_ids = [e['rule_id'] for e in result]
        corrections = [e['replacement'] for e in result]

        if len(result) > 0:
            detected += 1
            details.append({
                'id': item['id'],
                'status': 'DETECTED',
                'category': item['category'],
                'input': item['mistake_text'][:80],
                'corrections': corrections[:3],
                'rules': rule_ids[:3],
            })
        else:
            cat = item['category']
            missed_by_cat[cat] = missed_by_cat.get(cat, 0) + 1
            missed_ids.append(item['id'])
            details.append({
                'id': item['id'],
                'status': 'MISSED',
                'category': item['category'],
                'input': item['mistake_text'][:80],
                'expected': item['correct_text'][:80],
            })

    elapsed = time.time() - start
    total = len(dataset)
    rate = detected / total * 100 if total > 0 else 0

    print(f"{'='*70}")
    print(f"1000-SENTENCE DATASET EVALUATION")
    print(f"{'='*70}")
    print(f"Total sentences: {total}")
    print(f"Detected: {detected}/{total} ({rate:.1f}%)")
    print(f"Missed: {len(missed_ids)}/{total}")
    print(f"Time: {elapsed:.1f}s ({total/elapsed:.0f} sentences/sec)")
    print()

    print(f"=== MISSES BY CATEGORY ===")
    for cat, count in sorted(missed_by_cat.items(), key=lambda x: -x[1]):
        print(f"  {cat:30s}: {count}")
    print()

    print(f"=== MISSED SENTENCES (sample up to 50) ===")
    miss_count = 0
    for d in details:
        if d['status'] == 'MISSED' and miss_count < 50:
            print(f"  #{d['id']:3d} [{d['category']:25s}]")
            print(f"         Input:   {d['input']}")
            print(f"         Expected: {d['expected']}")
            print()
            miss_count += 1

    return detected, total, missed_ids, details

if __name__ == '__main__':
    evaluate()
