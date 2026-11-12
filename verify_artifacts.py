"""Verify final result coverage, pairing, frozen settings and source integrity."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    names = (ROOT / "quant/code/ucr142.txt").read_text().splitlines()
    methods = {"baseline", "uniform_half", "fisher_half", "pilot_half", "adaptive_pilot"}
    primary = pd.concat([pd.read_csv(ROOT / "reproduction/results" / f"{n}.csv") for n in names])
    extension = pd.concat([pd.read_csv(ROOT / "experiments/results" / f"{n}.csv") for n in names])
    assert len(primary) == 4260 and not primary.duplicated(["dataset", "resample"]).any()
    assert set(zip(primary.dataset, primary["resample"])) == {(n,r) for n in names for r in range(30)}
    assert len(extension) == 3550 and not extension.duplicated(["dataset", "resample", "method"]).any()
    assert set(zip(extension.dataset, extension["resample"], extension.method)) == {
        (n,r,m) for n in names for r in range(5) for m in methods}
    assert primary.accuracy.between(0,1).all() and extension.accuracy.between(0,1).all()
    assert np.isfinite(extension[["training_seconds", "predict_seconds", "accuracy"]]).all().all()
    assert (extension.training_seconds >= 0).all()
    assert np.allclose(extension.training_seconds, extension.selection_seconds+extension.fit_seconds)
    assert extension.selected_features.between(1, extension.original_features).all()
    baseline = extension[extension.method == "baseline"].merge(
        primary[["dataset", "resample", "accuracy"]], on=["dataset", "resample"],
        suffixes=("_extension", "_primary"), validate="one_to_one")
    assert len(baseline) == 710
    assert np.allclose(baseline.accuracy_extension, baseline.accuracy_primary, atol=1e-12, rtol=0)
    settings = json.loads((ROOT / "experiments/frozen_settings.json").read_text())
    assert settings["protocol_sha256"] == sha(ROOT / "experiments/PROTOCOL.md")
    assert settings["validation_sha256"] == sha(ROOT / "experiments/development/validation.csv")
    config = json.loads((ROOT / "experiments/results/configuration.json").read_text())
    assert config["feature_budget_sha256"] == sha(ROOT / "experiments/feature_budget.py")
    assert config["settings_sha256"] == sha(ROOT / "experiments/frozen_settings.json")
    dev = pd.read_csv(ROOT / "experiments/development/validation.csv")
    assert len(dev) == 504 and not dev.duplicated(["dataset", "fold", "method"]).any()
    diagnostic = ROOT / "reproduction/label_order_diagnostic"
    dconfig = json.loads((diagnostic / "configuration.json").read_text())
    assert len(dconfig["datasets"]) == 10
    for name in dconfig["datasets"]:
        df = pd.read_csv(diagnostic / f"{name}.csv")
        assert len(df) == 30 and set(df["resample"]) == set(range(30))
    index_audit = pd.read_csv(diagnostic / "index_audit.csv")
    assert len(index_audit) == 870 and index_audit.official_equals_lexical_set.all()
    checked_sources = []
    for relative, expected in json.loads((ROOT / "reproduction/source_checksums.json").read_text()).items():
        path = ROOT / relative
        if path.suffix == ".pdf" and not path.exists():
            continue  # The user's reference PDF is intentionally not redistributed.
        assert sha(path) == expected, f"Original source changed: {relative}"
        checked_sources.append(relative)
    validation = json.loads((ROOT / "reproduction/validation.json").read_text())
    for row in validation:
        assert all(row[key] for key in ["loader_equal", "2023_2025_features_equal",
            "cached_batched_features_allclose", "resample_predictions_equal"])
    output = dict(status="PASS", primary_rows=len(primary), extension_rows=len(extension),
        matching_baselines=len(baseline), development_rows=len(dev),
        historical_diagnostic_fits=290, historical_diagnostic_original_rows_reused=10,
        official_index_membership_checks=len(index_audit), checked_original_sources=checked_sources,
        protocol_and_settings_hashes_valid=True, recorded_numerical_validation_datasets=len(validation))
    (ROOT / "verification.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
