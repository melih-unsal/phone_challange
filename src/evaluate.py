"""Evaluate predictions against ground truth.

Two improvements over the original `evaluate.py`:

1. Multi-acceptable values: the README permits ground-truth fields to be a
   list of acceptable values (e.g. `"first_name": ["Lisa Marie", "Lisa-Marie"]`).
   We accept any of them.

2. Field-level normalised matching via `src.validators.field_match`:
     - emails are lowercased
     - phone numbers are E.164-parsed before comparison
     - names are NFC + casefolded + whitespace-collapsed
   This stops the pipeline losing points to formatting variations that are
   semantically equivalent.
"""

import json
import sys
from pathlib import Path

from src.validators import field_match


def evaluate(results_path: str = "results.json", ground_truth_path: str = "data/ground_truth.json") -> dict:
    with open(ground_truth_path, "r") as f:
        ground_truth = json.load(f)["recordings"]
    with open(results_path, "r") as f:
        result = json.load(f)["recordings"]

    if len(ground_truth) != len(result):
        print(f"WARNING: {len(ground_truth)} ground-truth recordings vs {len(result)} result recordings")

    ids = set(r["id"] for r in ground_truth) | set(r["id"] for r in result)

    mismatches: list[dict] = []
    per_key_total: dict[str, int] = {}
    per_key_correct: dict[str, int] = {}
    full_match = 0
    counted = 0

    for id in sorted(ids):
        gt_rec = next((r for r in ground_truth if r["id"] == id), None)
        res_rec = next((r for r in result if r["id"] == id), None)

        if gt_rec is None:
            print(f"Recording with id {id} is missing in ground truth")
            continue
        if res_rec is None:
            print(f"Recording with id {id} is missing in result")
            continue

        gt_info = gt_rec["expected"]
        res_info = res_rec["expected"]

        all_match = True
        counted += 1
        for key in gt_info.keys():
            gt_value = gt_info[key]
            res_value = res_info.get(key, "")
            per_key_total[key] = per_key_total.get(key, 0) + 1
            if field_match(res_value, gt_value, key=key):
                per_key_correct[key] = per_key_correct.get(key, 0) + 1
            else:
                all_match = False
                # Render the GT side as either the string or "<one of [...]>"
                gt_display = gt_value if not isinstance(gt_value, list) else " | ".join(gt_value)
                msg = f"Recording {id}: Mismatch in key '{key}' - expected: '{gt_display}', got: '{res_value}'"
                print(msg)
                mismatches.append({
                    "id": id,
                    "key": key,
                    "expected": gt_value,
                    "got": res_value,
                })
        if all_match:
            full_match += 1

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Recordings evaluated:  {counted}")
    print(f"Full-record matches:   {full_match} / {counted}")
    for key in sorted(per_key_total):
        c = per_key_correct.get(key, 0)
        t = per_key_total[key]
        pct = (100.0 * c / t) if t else 0.0
        print(f"  {key:14s} {c:>3} / {t:<3}  ({pct:5.1f}%)")

    return {
        "recordings_evaluated": counted,
        "full_match": full_match,
        "per_key_total": per_key_total,
        "per_key_correct": per_key_correct,
        "mismatches": mismatches,
    }


if __name__ == "__main__":
    results = sys.argv[1] if len(sys.argv) > 1 else "results.json"
    gt = sys.argv[2] if len(sys.argv) > 2 else "data/ground_truth.json"
    if not Path(results).exists():
        print(f"results file not found: {results}")
        sys.exit(1)
    evaluate(results, gt)
