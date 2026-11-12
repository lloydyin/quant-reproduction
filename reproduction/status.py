"""Read saved progress without importing ML libraries or interrupting training."""
import argparse
import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--results", type=Path, default=root / "results")
args = parser.parse_args()
names = (root.parent / "quant/code/ucr142.txt").read_text().splitlines()
counts, latest, latest_time = {}, None, -1
for name in names:
    path = args.results / f"{name}.csv"
    if path.exists():
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        completed = {int(row["resample"]) for row in rows if row.get("accuracy")}
        counts[name] = len(completed)
        if path.stat().st_mtime > latest_time:
            latest, latest_time = name, path.stat().st_mtime
completed_runs = sum(counts.values())
status = {"completed_runs": completed_runs, "expected_runs": 4260,
          "percent": round(100 * completed_runs / 4260, 2),
          "datasets_started": len(counts), "datasets_completed": sum(n == 30 for n in counts.values()),
          "latest_dataset": latest, "latest_dataset_runs": counts.get(latest, 0),
          "incomplete_datasets": {n: count for n, count in counts.items() if count < 30},
          "complete": completed_runs == 4260 and all(counts.get(n) == 30 for n in names)}
print(json.dumps(status, indent=2))
