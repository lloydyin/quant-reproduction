"""Select the proposed feature-budget setting using ORIGINAL TRAIN labels only."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import csv
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from feature_budget import forest, pilot_importance, select

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / "reproduction"))
from run_reproduction import digest


def choose_datasets():
    eligible, values = [], []
    names = (PROJECT / "quant/code/ucr142.txt").read_text().splitlines()
    for name in names:
        folder = PROJECT / "reproduction/features" / name
        meta = json.loads((folder / "metadata.json").read_text())
        labels = np.load(folder / "labels.npy")[:meta["n_train"]]
        if len(labels) >= 100 and np.bincount(labels).min() >= 5:
            eligible.append(name)
            values.append([np.log1p(meta[k]) for k in ("n_train", "length", "n_classes")])
    scaled = StandardScaler().fit_transform(values)
    model = KMeans(n_clusters=min(24, len(eligible)), random_state=0, n_init=10).fit(scaled)
    selected = []
    for cluster, centre in enumerate(model.cluster_centers_):
        candidates = np.flatnonzero(model.labels_ == cluster)
        closest = candidates[np.argmin(np.sum((scaled[candidates] - centre)**2, axis=1))]
        selected.append(eligible[closest])
    return sorted(selected)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-jobs", type=int, default=8)
    args = p.parse_args()
    out = ROOT / "development"
    out.mkdir(parents=True, exist_ok=True)
    names_path = out / "datasets.json"
    names = json.loads(names_path.read_text()) if names_path.exists() else choose_datasets()
    names_path.write_text(json.dumps(names, indent=2), encoding="utf-8")
    path = out / "validation.csv"
    done = set()
    if path.exists():
        df = pd.read_csv(path)
        done = set(zip(df.dataset, df.fold, df.method))
    fields = ["dataset", "fold", "method", "accuracy", "feature_fraction", "selection_seconds", "fit_seconds"]
    methods = ["baseline", "uniform_half", "fisher_half", "pilot_half", "adaptive_0.90", "adaptive_0.95", "adaptive_0.99"]
    for position, name in enumerate(names, 1):
        cache = PROJECT / "reproduction/features" / name
        meta = json.loads((cache / "metadata.json").read_text())
        z = np.load(cache / "features.npy", mmap_mode="r")
        y = np.load(cache / "labels.npy")[:meta["n_train"]]
        folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=2026)
        for fold, (train, valid) in enumerate(folds.split(np.zeros(len(y)), y)):
            pending = [m for m in methods if (name, fold, m) not in done]
            if not pending:
                continue
            x = np.asarray(z[train], order="F")
            pilot = pilot_importance(x, y[train], fold, args.n_jobs)
            for method in pending:
                adaptive = method.startswith("adaptive_")
                mass = float(method.split("_")[1]) if adaptive else 0.95
                columns, selection_seconds, _ = select("adaptive_pilot" if adaptive else method,
                    x, y[train], fold, args.n_jobs, mass, pilot=pilot)
                start = perf_counter()
                model = forest(fold, args.n_jobs).fit(np.asarray(x[:, columns], order="F"), y[train])
                fit_seconds = perf_counter() - start
                predictions = model.predict(z[np.ix_(valid, columns)])
                row = dict(dataset=name, fold=fold, method=method,
                           accuracy=float(np.mean(predictions == y[valid])), feature_fraction=len(columns)/z.shape[1],
                           selection_seconds=selection_seconds, fit_seconds=fit_seconds)
                new = not path.exists()
                with path.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    if new: writer.writeheader()
                    writer.writerow(row)
                print(f"[{position}/{len(names)}] {name} fold={fold} {method}: {row['accuracy']:.4f} fraction={row['feature_fraction']:.3f}", flush=True)
                del model
            del x
    df = pd.read_csv(path)
    expected = len(names)*3*len(methods)
    if len(df) != expected or df.duplicated(["dataset", "fold", "method"]).any():
        raise ValueError(f"Incomplete development: {len(df)} vs {expected}")
    table = df.groupby(["dataset", "method"]).mean(numeric_only=True).groupby("method").mean()
    table.to_csv(out / "summary.csv")
    adaptive = table.loc[table.index.str.startswith("adaptive_")]
    contenders = adaptive[adaptive.accuracy >= adaptive.accuracy.max()-0.005]
    chosen = contenders.sort_values(["feature_fraction"], kind="stable").index[0]
    settings = {"mass": float(chosen.split("_")[1]), "pilot_trees": 32, "coverage_reserve": 0.2,
                "min_fraction": 0.25, "max_fraction": 0.75, "final_trees": 200,
                "selection_rule": "within 0.005 of best validation accuracy, then minimum feature fraction",
                "development_datasets": names, "development_folds": 3,
                "development_data": "original TRAIN only", "protocol_sha256": digest(ROOT / "PROTOCOL.md"),
                "validation_sha256": digest(path)}
    (ROOT / "frozen_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(json.dumps(settings, indent=2))


if __name__ == "__main__":
    main()
