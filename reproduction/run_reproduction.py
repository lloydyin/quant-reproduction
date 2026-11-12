"""Reproduce QUANT accuracy with official splits and the unchanged author code.

The Quant transform is sample-wise and fitted only to series length. We cache
its output to avoid computing identical features 30 times. Timings for feature
preparation are recorded separately and are NOT paper-comparable benchmark times.
"""
import argparse
import csv
import gc
import hashlib
import io
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import re
import sys
from time import perf_counter
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / "quant" / "code"))

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
TORCH_VERSION = version("torch")


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def key(name):
    return re.sub(r"[^a-z0-9]", "", name.lower()).replace("discrete", "disc")


def load_ts(path):
    """Strict loader for this benchmark's equal-length, univariate .ts files."""
    rows, labels, in_data = [], [], False
    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if not in_data:
                if line.lower().startswith("@timestamps true"):
                    raise ValueError(f"Timestamped data unsupported: {path}")
                in_data = line.lower() == "@data"
                continue
            parts = line.split(":")
            if len(parts) != 2:
                raise ValueError(f"Expected univariate labelled data: {path}")
            values = np.fromstring(parts[0], sep=",", dtype=np.float64).astype(np.float32)
            if len(values) != len(parts[0].split(",")) or not np.isfinite(values).all():
                raise ValueError(f"Missing or invalid values: {path}")
            rows.append(values)
            labels.append(parts[1].strip().lower())
    if not rows or len({len(row) for row in rows}) != 1:
        raise ValueError(f"Empty or unequal-length data: {path}")
    return np.stack(rows)[:, None, :], np.asarray(labels)


def data_files(name):
    pair = [ROOT / "data" / name / f"{name}_{s}.ts" for s in ("TRAIN", "TEST")]
    if all(p.exists() for p in pair):
        return pair
    result = []
    for split in ("TRAIN", "TEST"):
        matches = [p for p in (ROOT / "data_extra").rglob("*.ts")
                   if key(p.stem) == key(f"{name}_{split}")]
        if len(matches) != 1:
            raise FileNotFoundError(f"{name} {split}: expected one source, found {matches}")
        result.append(matches[0])
    return result


def read_indices(name, resample, source, n_train, n_total, labels):
    if resample == 0:
        return np.arange(n_train), np.arange(n_train, n_total), "original"
    archive = ROOT / "downloads" / f"{source}_indices.zip"
    with zipfile.ZipFile(archive) as z:
        matches = [p for p in z.namelist() if
                   key(Path(p).parent.name) == key(name)
                   and Path(p).name == f"resample{resample}Indices_TRAIN.txt"]
        if len(matches) != 1:
            raise ValueError(f"Missing/ambiguous official indices: {name}, {resample}, {source}")
        train = np.loadtxt(io.BytesIO(z.read(matches[0])), dtype=np.int64, ndmin=1)
        test = np.setdiff1d(np.arange(n_total), train)
        # Follow the author's run.py: test indices are the sorted complement.
    if len(train) != n_train or len(np.unique(train)) != n_train or train.min() < 0 or train.max() >= n_total:
        raise ValueError(f"Invalid or incompatible indices: {name}/{resample}")
    if not np.array_equal(np.bincount(labels[train]), np.bincount(labels[:n_train])):
        raise ValueError(f"Resample class counts differ: {name}/{resample}")
    return train, test, source


def prepare(name, paths, args, source_hash):
    cache = ROOT / "features" / name
    cache.mkdir(parents=True, exist_ok=True)
    config = {"source_sha256": source_hash, "depth": 6, "div": 4,
              "torch": TORCH_VERSION, "numpy": np.__version__,
              "files": {str(p.relative_to(ROOT)): digest(p) for p in paths}}
    meta_path = cache / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["config"] == config and (cache / "features.npy").exists():
            return np.load(cache / "features.npy", mmap_mode="r"), np.load(cache / "labels.npy"), meta
    # The complete cache lets the long training stage avoid loading PyTorch.
    import torch
    from quant import Quant
    torch.set_num_threads(1)
    started = perf_counter()
    x_train, y_train = load_ts(paths[0])
    x_test, y_test = load_ts(paths[1])
    if x_train.shape[1:] != x_test.shape[1:]:
        raise ValueError(f"Train/test shapes differ: {name}")
    le = LabelEncoder().fit(y_train)
    labels = np.concatenate([le.transform(y_train), le.transform(y_test)])
    count, n_train = len(labels), len(y_train)
    quant, offset, feature_array = Quant(depth=6, div=4), 0, None
    transform_seconds = []
    with torch.inference_mode():
        for x in (x_train, x_test):
            transform_start = perf_counter()
            for start in range(0, len(x), args.batch_size):
                batch = torch.from_numpy(x[start:start + args.batch_size])
                z = quant.transform(batch) if quant.fitted else quant.fit_transform(batch, None)
                if feature_array is None:
                    feature_array = np.lib.format.open_memmap(cache / "features.tmp.npy", mode="w+", dtype="float32", shape=(count, z.shape[1]))
                feature_array[offset:offset + len(batch)] = z.numpy()
                offset += len(batch)
            transform_seconds.append(perf_counter() - transform_start)
    feature_array.flush()
    features = feature_array.shape[1]
    del feature_array, x_train, x_test, z, batch
    (cache / "features.tmp.npy").replace(cache / "features.npy")
    np.save(cache / "labels.npy", labels)
    meta = {"config": config, "n_train": n_train, "n_test": count - n_train,
            "length": quant.models[0].intervals[-1, 1].item(),
            "n_features": features, "n_classes": len(le.classes_),
            "classes": le.classes_.tolist(), "batch_size": args.batch_size,
            "prepare_seconds": perf_counter() - started,
            "transform_train_batched_seconds": transform_seconds[0],
            "transform_test_batched_seconds": transform_seconds[1]}
    # The first interval always spans the complete original series.
    meta["length"] = int(quant.models[0].intervals[0, 1])
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    gc.collect()
    return np.load(cache / "features.npy", mmap_mode="r"), labels, meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=30)
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--output", type=Path, default=ROOT / "results")
    args = p.parse_args()
    if not 0 <= args.start < args.stop <= 30:
        p.error("Require 0 <= start < stop <= 30")
    all_names = (PROJECT / "quant" / "code" / "ucr142.txt").read_text().splitlines()
    names = args.datasets or all_names
    if set(names) - set(all_names):
        p.error("Dataset names must appear in the author's ucr142.txt")
    args.output.mkdir(parents=True, exist_ok=True)
    source_hash = digest(PROJECT / "quant" / "code" / "quant.py")
    environment = {"python": sys.version, "platform": platform.platform(),
                   "torch": TORCH_VERSION, "numpy": np.__version__,
                   "scikit_learn": sklearn.__version__, "quant_sha256": source_hash,
                   "n_estimators": 200, "max_features": 0.1, "criterion": "entropy",
                   "depth": 6, "div": 4, "random_state": "resample index (0..29)",
                   "n_jobs": args.n_jobs, "torch_threads": 1,
                   "indices": "Java for original 112, Python for additional 30",
                   "index_archive_sha256": {s: digest(ROOT / "downloads" / f"{s}_indices.zip") for s in ["java", "python"]},
                   "author_results_sha256": digest(PROJECT / "quant/results/accuracy_resamples_ucr142.csv"),
                   "feature_cache": "sample-wise, independent of labels and other cases"}
    env_path = args.output / "environment.json"
    if env_path.exists() and json.loads(env_path.read_text()) != environment:
        raise ValueError("Output contains a different experiment configuration; choose another --output")
    env_path.write_text(json.dumps(environment, indent=2), encoding="utf-8")
    fields = ["dataset", "resample", "accuracy", "paper_accuracy", "difference",
              "fit_seconds", "predict_seconds", "n_train", "n_test", "length",
              "n_features", "n_classes", "random_state", "indices_source"]
    paper = pd.read_csv(PROJECT / "quant" / "results" / "accuracy_resamples_ucr142.csv", index_col=0)
    failures = []
    for number, name in enumerate(names, 1):
        try:
            path = args.output / f"{name}.csv"
            done = set(pd.read_csv(path)["resample"].astype(int)) if path.exists() else set()
            remaining = [r for r in range(args.start, args.stop) if r not in done]
            if not remaining:
                continue
            z, labels, meta = prepare(name, data_files(name), args, source_hash)
            source = "java" if name in all_names[:112] else "python"
            for resample in remaining:
                train, test, used_source = read_indices(name, resample, source, meta["n_train"], len(labels), labels)
                classifier = ExtraTreesClassifier(n_estimators=200, max_features=0.1,
                    criterion="entropy", n_jobs=args.n_jobs, random_state=resample)
                train_z = np.asarray(z[train], order="F")
                start = perf_counter()
                classifier.fit(train_z, labels[train])
                fit_seconds = perf_counter() - start
                del train_z
                start = perf_counter()
                predictions = np.concatenate([classifier.predict(z[test[i:i + 1024]]) for i in range(0, len(test), 1024)])
                predict_seconds = perf_counter() - start
                accuracy = float(np.mean(predictions == labels[test]))
                reference = float(paper.loc[name, str(resample)])
                row = dict(dataset=name, resample=resample, accuracy=accuracy, paper_accuracy=reference,
                           difference=accuracy-reference, fit_seconds=fit_seconds, predict_seconds=predict_seconds,
                           **{k: meta[k] for k in ["n_train", "n_test", "length", "n_features", "n_classes"]},
                           random_state=resample, indices_source=used_source)
                new = not path.exists()
                with path.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    if new:
                        writer.writeheader()
                    writer.writerow(row)
                print(f"[{number}/{len(names)}] {name} r={resample:02d} accuracy={accuracy:.6f} paper={reference:.6f} fit={fit_seconds:.2f}s", flush=True)
                del classifier
            del z, labels
            gc.collect()
        except Exception as exc:
            failures.append({"dataset": name, "error": repr(exc)})
            print(f"FAILED {name}: {exc!r}", flush=True)
    (args.output / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    if failures:
        raise SystemExit(f"{len(failures)} datasets need attention; see failures.json")


if __name__ == "__main__":
    main()
