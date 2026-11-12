"""Generate auditable result tables and figures, including partial-run coverage."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT.parent / ".cache" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=ROOT / "results")
    args = p.parse_args()
    names = (ROOT.parent / "quant/code/ucr142.txt").read_text().splitlines()
    frames = [pd.read_csv(args.results / f"{n}.csv") for n in names if (args.results / f"{n}.csv").exists()]
    if not frames:
        raise SystemExit("No completed results yet")
    runs = pd.concat(frames, ignore_index=True).sort_values(["dataset", "resample"])
    if runs.duplicated(["dataset", "resample"]).any():
        raise ValueError("Duplicate dataset/resample rows")
    runs.to_csv(args.results / "all_runs.csv", index=False)
    summary = runs.groupby("dataset").agg(
        resamples=("resample", "nunique"), reproduced_accuracy=("accuracy", "mean"),
        reproduced_std=("accuracy", "std"), paper_same_resamples=("paper_accuracy", "mean"),
        mean_fit_seconds=("fit_seconds", "mean"), mean_predict_seconds=("predict_seconds", "mean"),
        n_train=("n_train", "first"), n_test=("n_test", "first"), length=("length", "first"),
        n_features=("n_features", "first"), n_classes=("n_classes", "first"))
    published = pd.read_csv(ROOT.parent / "quant/results/accuracy_mean_ucr142.csv", index_col=0)
    metadata_snapshot = ROOT / "dataset_metadata.json"
    preserved = json.loads(metadata_snapshot.read_text()) if metadata_snapshot.exists() else {}
    catalog = []
    for name in names:
        item = {"dataset": name, "group": "UCR112" if name in names[:112] else "Additional30",
                "author_mean_accuracy_30_resamples": float(published.loc[name, "accuracy"])}
        meta_path = ROOT / "features" / name / "metadata.json"
        if meta_path.exists() or name in preserved:
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else preserved[name]
            item.update({k: meta[k] for k in ["n_train", "n_test", "length", "n_classes", "n_features"]})
            item["data_files"] = ";".join(meta["config"]["files"])
        catalog.append(item)
    pd.DataFrame(catalog).to_csv(ROOT / "dataset_catalog.csv", index=False)
    summary["paper_30_resample_accuracy"] = published.accuracy
    summary["difference_same_resamples"] = summary.reproduced_accuracy - summary.paper_same_resamples
    summary["difference_percentage_points"] = 100 * summary.difference_same_resamples
    summary["group"] = ["UCR112" if n in names[:112] else "Additional30" for n in summary.index]
    summary.to_csv(args.results / "comparison.csv")
    complete = summary[summary.resamples == 30]
    full = len(complete) == 142 and len(runs) == 4260
    stats = {"datasets_with_results": len(summary), "completed_runs": len(runs), "expected_runs": 4260,
             "datasets_with_30_resamples": len(complete), "full_reproduction_complete": full,
             "missing_datasets": sorted(set(names) - set(summary.index)),
             "minimum_resamples": int(summary.resamples.min()), "maximum_resamples": int(summary.resamples.max()),
             "macro_accuracy_available": float(summary.reproduced_accuracy.mean()),
             "paper_matched_macro_accuracy": float(summary.paper_same_resamples.mean()),
             "paper_all142_30resamples_macro_accuracy": float(published.accuracy.mean())}
    comparison_rows = []
    corrected = {"Covid3MonthDiscrete", "FloodModeling1Discrete", "FloodModeling2Discrete", "FloodModeling3Discrete"}
    summary["known_data_correction"] = summary.index.isin(corrected)
    summary.to_csv(args.results / "comparison.csv")
    for label, part in [("All available", summary), ("Excluding 4 corrected datasets", summary[~summary.known_data_correction]),
                        ("Original 112", summary[summary.group == "UCR112"]),
                        ("Additional 30", summary[summary.group == "Additional30"])]:
        if part.empty:
            continue
        differences = part.difference_same_resamples.to_numpy()
        differences = np.round(differences, 6)
        pvalue = float(wilcoxon(differences).pvalue) if np.any(differences) else 1.0
        item = {"group": label, "datasets": len(part), "ours": float(part.reproduced_accuracy.mean()),
                "paper_matched": float(part.paper_same_resamples.mean()),
                "difference_pp": float(100 * differences.mean()), "wilcoxon_p": pvalue,
                "wins": int((differences > 0).sum()), "ties": int((differences == 0).sum()),
                "losses": int((differences < 0).sum())}
        comparison_rows.append(item)
    stats["groups"] = comparison_rows
    (args.results / "summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(6, 6))
    for group, colour in [("UCR112", "#2864b7"), ("Additional30", "#df7728")]:
        part = summary[(summary.group == group) & ~summary.known_data_correction]
        ax.scatter(part.paper_same_resamples, part.reproduced_accuracy, label=group, s=25, alpha=.75, color=colour)
    changed = summary[summary.known_data_correction]
    if not changed.empty:
        ax.scatter(changed.paper_same_resamples, changed.reproduced_accuracy,
                   label="Known data corrections", color="#bd3030", marker="x", s=48)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set(xlim=(0, 1.01), ylim=(0, 1.01), xlabel="Author accuracy (matched resamples)", ylabel="Reproduced accuracy",
           title=f"QUANT reproduction: {len(summary)} datasets, {len(runs)}/4260 runs")
    ax.legend()
    ax.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(args.results / "accuracy_comparison.png", dpi=180)
    fig.savefig(args.results / "accuracy_comparison.pdf")
    plt.close(fig)
    lines = ["# QUANT reproduction results", "", f"Completed {len(runs)}/4260 runs across {len(summary)}/142 datasets; {len(complete)} datasets have all 30 resamples.",
             "", "Status: " + ("The 142 x 30 primary accuracy reproduction is complete." if full else "Partial results; the primary reproduction is not complete."), "",
             "The table uses equal dataset weights and matches the author's resample columns. Unequal completion counts indicate a progress snapshot only. Accuracy is expressed as a fraction; differences are percentage points.", "",
             "| Scope | Datasets | Reproduced accuracy | Author accuracy | Difference (pp) | Wilcoxon p |", "|---|---:|---:|---:|---:|---:|"]
    for item in comparison_rows:
        lines.append(f"| {item['group']} | {item['datasets']} | {item['ours']:.6f} | {item['paper_matched']:.6f} | {item['difference_pp']:+.4f} | {item['wilcoxon_p']:.6g} |")
    lines += ["", "## Fifteen largest absolute dataset differences", "", "| Dataset | Resamples | Reproduced | Author | Difference (pp) |", "|---|---:|---:|---:|---:|"]
    for name, row in summary.loc[summary.difference_same_resamples.abs().sort_values(ascending=False).index].head(15).iterrows():
        lines.append(f"| {name} | {int(row.resamples)} | {row.reproduced_accuracy:.6f} | {row.paper_same_resamples:.6f} | {row.difference_percentage_points:+.4f} |")
    lines += ["", "## Interpretation and scope", "", "- The author did not fix the ExtraTrees seed; this run uses random_state=resample.",
              "- The local quant/code/quant.py is unchanged. The 2023 implementation and 2025 multivariate update produced identical features on the four checked univariate datasets.",
              "- Features are cached in batches using a sample-wise transform. Checked floating-point differences were at most approximately 3.8e-6, with identical seeded predictions.",
              "- Caching and multithreaded classifiers change the computation schedule. These timings do not reproduce the paper's single-core, freshly transformed pipeline; see environment.json for thread settings.",
              "- The expanded benchmark documents corrections to four datasets; see the README. Do not automatically attribute data-version differences to the model.",
              "- Paired two-sided Wilcoxon tests use dataset-mean accuracy differences rounded to six decimal places. Their p-values do not establish equivalence.",
              "- This study reproduces QUANT's primary accuracy experiment. Competing models and the paper's full sensitivity study were not rerun.", "",
              "Per-split records: all_runs.csv. Dataset comparisons: comparison.csv. Execution environment: environment.json."]
    (args.results / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
