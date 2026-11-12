"""Summarise the paired QUANT extension evaluation and render report figures."""
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT.parent / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = ["baseline", "uniform_half", "fisher_half", "pilot_half", "adaptive_pilot"]
LABELS = {"baseline": "QUANT", "uniform_half": "Uniform half", "fisher_half": "Fisher half",
          "pilot_half": "Pilot half", "adaptive_pilot": "Adaptive pilot"}


def holm(values):
    values = np.asarray(values)
    order = np.argsort(values)
    adjusted, previous = np.empty(len(values)), 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, values[index] * (len(values)-rank)))
        adjusted[index] = previous
    return adjusted


def main():
    out = ROOT / "results"
    names = (ROOT.parent / "quant/code/ucr142.txt").read_text().splitlines()
    frames = [pd.read_csv(out / f"{n}.csv") for n in names if (out / f"{n}.csv").exists()]
    if not frames: raise SystemExit("No extension results")
    runs = pd.concat(frames, ignore_index=True)
    if runs.duplicated(["dataset", "resample", "method"]).any():
        raise ValueError("Duplicated experiment rows")
    runs.to_csv(out / "all_runs.csv", index=False)
    complete_names = []
    for name, part in runs.groupby("dataset"):
        if len(part) == 25 and set(zip(part["resample"], part.method)) == {(r,m) for r in range(5) for m in METHODS}:
            complete_names.append(name)
    if not complete_names: raise SystemExit("No dataset has all five methods and five resamples")
    complete = runs[runs.dataset.isin(complete_names)]
    means = complete.groupby(["dataset", "method"]).mean(numeric_only=True)
    means.to_csv(out / "dataset_means.csv")
    baseline = means.xs("baseline", level="method")
    metrics, pvalues, differences = [], [], []
    for method in METHODS:
        part = means.xs(method, level="method").loc[baseline.index]
        delta = np.round((part.accuracy-baseline.accuracy).to_numpy(), 12)
        p = float(wilcoxon(delta).pvalue) if np.any(delta) else 1.0
        if method != "baseline": pvalues.append(p)
        metric = {"method": method, "datasets": len(part), "mean_accuracy": float(part.accuracy.mean()),
                  "median_accuracy": float(part.accuracy.median()), "accuracy_difference_pp": float(100*delta.mean()),
                  "median_accuracy_difference_pp": float(100*np.median(delta)),
                  "wins": int((delta>0).sum()), "ties": int((delta==0).sum()), "losses": int((delta<0).sum()),
                  "wilcoxon_p": p, "mean_feature_fraction": float(part.feature_fraction.mean()),
                  "median_feature_fraction": float(part.feature_fraction.median()),
                  "total_training_seconds_per_resample": float(part.training_seconds.sum()),
                  "total_training_ratio": float(part.training_seconds.sum()/baseline.training_seconds.sum()),
                  "median_training_ratio": float((part.training_seconds/baseline.training_seconds).median()),
                  "total_prediction_ratio": float(part.predict_seconds.sum()/baseline.predict_seconds.sum()),
                  "total_selected_matrix_ratio": float(part.selected_matrix_bytes.sum()/baseline.selected_matrix_bytes.sum()),
                  "total_tree_array_ratio": float(part.tree_array_bytes.sum()/baseline.tree_array_bytes.sum()),
                  "mean_selection_seconds": float(part.selection_seconds.mean())}
        metrics.append(metric)
        differences.append(pd.DataFrame({"dataset": part.index, "method": method,
                                        "accuracy_difference_pp": 100*delta,
                                        "feature_fraction": part.feature_fraction.to_numpy(),
                                        "training_ratio": (part.training_seconds/baseline.training_seconds).to_numpy()}))
    corrections = holm(pvalues)
    metrics[0]["holm_p"] = 1.0
    for metric, corrected in zip(metrics[1:], corrections): metric["holm_p"] = float(corrected)
    table = pd.DataFrame(metrics).set_index("method")
    table.to_csv(out / "summary.csv")
    pd.concat(differences, ignore_index=True).to_csv(out / "paired_differences.csv", index=False)
    original_rows = []
    for name in complete_names:
        original_rows.append(pd.read_csv(ROOT.parent / "reproduction/results" / f"{name}.csv").query("resample < 5"))
    original = pd.concat(original_rows, ignore_index=True)
    check = complete[complete.method == "baseline"].merge(original[["dataset", "resample", "accuracy"]],
        on=["dataset", "resample"], suffixes=("_paired", "_reproduction"), validate="one_to_one")
    max_diff = float(np.max(np.abs(check.accuracy_paired-check.accuracy_reproduction)))
    result = {"complete_datasets": len(complete_names), "completed_rows": len(runs), "expected_rows": 3550,
              "full_evaluation_complete": len(complete_names)==142 and len(runs)==3550,
              "baseline_agreement_max_absolute_accuracy_difference": max_diff,
              "baseline_agreement_count": int(np.isclose(check.accuracy_paired,check.accuracy_reproduction,atol=1e-12,rtol=0).sum()),
              "baseline_agreement_expected": len(check), "methods": metrics}
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.25))
    order = METHODS[1:]
    labels = [LABELS[m] for m in order]
    colours = ["#8b9dac", "#d6943c", "#507db7", "#2b8d72"]
    for ax, column, title, reference in zip(axes,
        ["accuracy_difference_pp", "total_training_ratio", "mean_feature_fraction"],
        ["Accuracy change (pp)", "Downstream training time ratio", "Mean retained feature fraction"], [0,1,1]):
        ax.barh(labels, table.loc[order,column], color=colours)
        ax.axvline(reference, color="black", linestyle="--", linewidth=.8)
        ax.set_title(title)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.15)
    fig.tight_layout()
    for ext in ("png","pdf"): fig.savefig(out / f"tradeoffs.{ext}", dpi=200)
    plt.close(fig)

    adaptive = means.xs("adaptive_pilot", level="method").loc[baseline.index]
    fig, axes = plt.subplots(1, 2, figsize=(7.0,3.0))
    axes[0].scatter(baseline.accuracy, adaptive.accuracy, s=14, alpha=.7, color="#2b8d72")
    axes[0].plot([0,1],[0,1],"--",color="gray",linewidth=.8)
    axes[0].set(xlabel="QUANT accuracy",ylabel="Adaptive pilot accuracy",xlim=(0,1.01),ylim=(0,1.01))
    points=axes[1].scatter(adaptive.length,adaptive.feature_fraction,c=np.log10(adaptive.n_train),cmap="viridis",s=19)
    axes[1].set(xscale="log",xlabel="Series length",ylabel="Retained feature fraction",ylim=(0.2,.8))
    fig.colorbar(points,ax=axes[1],label="log10(training cases)")
    fig.tight_layout()
    for ext in ("png","pdf"): fig.savefig(out / f"adaptive_detail.{ext}", dpi=200)
    plt.close(fig)

    lines=["# Feature-budget experiment results", "", f"Completed all methods and splits for {len(complete_names)}/142 datasets, with {len(runs)}/3550 method records. Each method uses the same five official splits.", "",
           "Accuracy and feature retention are fractions. Accuracy differences are percentage points relative to the matched baseline.", "",
           "| Method | Mean accuracy | Difference (pp) | Wins/ties/losses | Holm p | Retained feature fraction | Downstream training time ratio |",
           "|---|---:|---:|---|---:|---:|---:|"]
    for item in metrics:
        lines.append(f"| {LABELS[item['method']]} | {item['mean_accuracy']:.6f} | {item['accuracy_difference_pp']:+.4f} | {item['wins']}/{item['ties']}/{item['losses']} | {item['holm_p']:.5g} | {item['mean_feature_fraction']:.3f} | {item['total_training_ratio']:.3f} |")
    lines += ["", "Training time includes feature selection, the pilot forest and final classifier fitting. It excludes the common QUANT feature transformation and data loading.",
              "Selected-matrix bytes and tree-array bytes are not peak process RAM. These measurements do not establish end-to-end acceleration or lower peak RAM.",
              f"Maximum baseline accuracy difference from the corresponding five primary-reproduction splits: {max_diff:.12g}.", "",
              "Settings were frozen after internal training-set validation. Unsuccessful and degraded variants are retained in the report."]
    (out / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
