"""Paired test evaluation of the frozen QUANT feature-budget variants.

Run AFTER the original reproduction and development have finished. Each method
is benchmarked in one process on the same five official splits. No test-set
feedback is used to choose the frozen setting.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
import zlib
import numpy as np
import pandas as pd
from feature_budget import forest, pilot_importance, select, tree_array_bytes

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / "reproduction"))
from run_reproduction import digest, read_indices

METHODS = ["baseline", "uniform_half", "fisher_half", "pilot_half", "adaptive_pilot"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--resamples", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    if not 1 <= args.resamples <= 30:
        parser.error("resamples must be 1..30")
    settings_path = ROOT / "frozen_settings.json"
    settings = json.loads(settings_path.read_text())
    if settings["protocol_sha256"] != digest(ROOT / "PROTOCOL.md"):
        raise ValueError("Protocol changed after development")
    all_names = (PROJECT / "quant/code/ucr142.txt").read_text().splitlines()
    names = args.datasets or all_names
    if set(names) - set(all_names):
        parser.error("Unknown dataset")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {"settings": settings, "settings_sha256": digest(settings_path),
              "feature_budget_sha256": digest(ROOT / "feature_budget.py"),
              "n_jobs": args.n_jobs, "methods": METHODS, "resamples": args.resamples,
              "numpy_version": np.__version__, "timing": "downstream selection + final fit; common feature extraction excluded",
              "pilot_charged_to_each_variant": True,
              "index_sha256": {s: digest(PROJECT / "reproduction/downloads" / f"{s}_indices.zip") for s in ["java", "python"]}}
    config_path = args.output / "configuration.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Existing output uses a different experiment configuration")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    fields = ["dataset", "resample", "method", "accuracy", "selection_seconds", "fit_seconds", "training_seconds",
              "predict_seconds", "original_features", "selected_features", "feature_fraction", "selected_matrix_bytes",
              "tree_array_bytes", "pilot_seconds", "importance_prefix_count", "retained_importance", "columns_sha256",
              "n_train", "n_test", "length", "n_classes", "indices_source", "seed"]
    failures = []
    for position, name in enumerate(names, 1):
        try:
            cache = PROJECT / "reproduction/features" / name
            meta = json.loads((cache / "metadata.json").read_text())
            z = np.load(cache / "features.npy", mmap_mode="r")
            y = np.load(cache / "labels.npy")
            output = args.output / f"{name}.csv"
            done = set()
            if output.exists():
                completed = pd.read_csv(output)
                done = set(zip(completed["resample"].astype(int), completed["method"]))
            for resample in range(args.resamples):
                pending = [m for m in METHODS if (resample, m) not in done]
                if not pending:
                    continue
                source = "java" if name in all_names[:112] else "python"
                train, test, used_source = read_indices(name, resample, source, meta["n_train"], len(y), y)
                x = np.asarray(z[train], order="F")
                rng = np.random.default_rng(zlib.crc32(name.encode()) + resample)
                order = rng.permutation(pending)
                pilot = None
                for method in order:
                    if method in ("pilot_half", "adaptive_pilot") and pilot is None:
                        pilot = pilot_importance(x, y[train], resample, args.n_jobs)
                    columns, selection_seconds, extra = select(method, x, y[train], resample,
                        args.n_jobs, settings["mass"], pilot=pilot)
                    start = perf_counter()
                    selected = x if method == "baseline" else np.asarray(x[:, columns], order="F")
                    model = forest(resample, args.n_jobs).fit(selected, y[train])
                    fit_seconds = perf_counter()-start
                    del selected
                    start = perf_counter()
                    predictions = np.concatenate([model.predict(z[np.ix_(test[i:i+1024], columns)]) for i in range(0, len(test), 1024)])
                    prediction_seconds = perf_counter()-start
                    accuracy = float(np.mean(predictions == y[test]))
                    row = dict(dataset=name, resample=resample, method=method, accuracy=accuracy,
                        selection_seconds=selection_seconds, fit_seconds=fit_seconds,
                        training_seconds=selection_seconds+fit_seconds, predict_seconds=prediction_seconds,
                        original_features=z.shape[1], selected_features=len(columns), feature_fraction=len(columns)/z.shape[1],
                        selected_matrix_bytes=len(train)*len(columns)*4, tree_array_bytes=tree_array_bytes(model),
                        pilot_seconds=extra.get("pilot_seconds", 0), importance_prefix_count=extra.get("importance_prefix_count", ""),
                        retained_importance=extra.get("retained_importance", ""),
                        columns_sha256=hashlib.sha256(columns.astype("<i8").tobytes()).hexdigest(),
                        **{k: meta[k] for k in ["n_train", "n_test", "length", "n_classes"]},
                        indices_source=used_source, seed=resample)
                    new = not output.exists()
                    with output.open("a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=fields)
                        if new: writer.writeheader()
                        writer.writerow(row)
                    print(f"[{position}/{len(names)}] {name} r={resample} {method}: accuracy={accuracy:.6f} features={len(columns)}/{z.shape[1]} train={row['training_seconds']:.3f}s", flush=True)
                    del model
                del x
                gc.collect()
            del z, y
        except Exception as exc:
            failures.append({"dataset": name, "error": repr(exc)})
            print(f"FAILED {name}: {exc!r}", flush=True)
    (args.output / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    if failures:
        raise SystemExit(f"{len(failures)} datasets failed")


if __name__ == "__main__":
    main()
