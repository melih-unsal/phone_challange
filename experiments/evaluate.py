import json
import sys
with open("data/ground_truth.json", "r") as f:
    ground_truth = json.load(f)["recordings"]

path = sys.argv[1] if len(sys.argv) > 1 else "results.json"

with open(path, "r") as f:
    result = json.load(f)["recordings"]   

assert len(ground_truth) == len(result), "Number of recordings in ground truth and result do not match"

ids = set(rec['id'] for rec in ground_truth) | set(rec['id'] for rec in result)

for id in ids:
    gt_rec = next((rec for rec in ground_truth if rec['id'] == id), None)
    res_rec = next((rec for rec in result if rec['id'] == id), None)
    
    if gt_rec is None:
        print(f"Recording with id {id} is missing in ground truth")
        continue
    if res_rec is None:
        print(f"Recording with id {id} is missing in result")
        continue
    
    gt_info = gt_rec["expected"]
    res_info = res_rec["expected"]
    
    for key in gt_info.keys():
        gt_value = gt_info[key]
        res_value = res_info.get(key, "")
        if gt_value != res_value:
            print(f"Recording {id}: Mismatch in key '{key}' - expected: '{gt_value}', got: '{res_value}'")