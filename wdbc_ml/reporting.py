"""Write portable tables, figures, and a readable experiment report."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def save_csv(frame, path, index=False):
    """Preserve near-equal probabilities; rounding can change AUC and AP ranks."""
    frame.to_csv(path, index=index, float_format="%.17g")


def markdown_table(frame):
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def save_plots(results, exploration, output):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 120})
    summary = results["summary"].sort_values("roc_auc_mean", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
    for ax, metric, title in zip(axes, ["roc_auc", "f1_malignant"],
                                 ["ROC AUC", "Malignant-class F1"]):
        ax.barh(summary.model.str.replace("_", " "), summary[f"{metric}_mean"],
                xerr=summary[f"{metric}_std"], color="#27758c", capsize=3)
        ax.set(xlim=(0, 1.02), xlabel="Outer-fold mean ± sample SD", title=title)
        ax.grid(axis="x", alpha=0.2)
    fig.savefig(output / "model_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.5), layout="constrained")
    for name, frame in results["predictions"].groupby("model", sort=True):
        fpr, tpr, _ = roc_curve(frame.y_true, frame.p_malignant)
        ax.plot(fpr, tpr, label=name.replace("_", " "))
    ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.6)
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="Pooled out-of-fold ROC curves", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(output / "roc_curves.png", dpi=160)
    plt.close(fig)

    assignments = exploration["assignments"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for ax, column in zip(axes.flat, ["malignant", "kmeans", "ward", "dbscan", "gaussian_mixture"]):
        for value in sorted(assignments[column].unique()):
            points = assignments[assignments[column] == value]
            label = ("malignant" if value == 1 else "benign") if column == "malignant" else str(value)
            color = "#999999" if value == -1 else ("#d47738" if value == 1 else "#27758c")
            ax.scatter(points.pc1, points.pc2, s=9, alpha=0.65, label=label, color=color)
        ax.set(title=column.replace("_", " "), xlabel="PC1", ylabel="PC2")
        ax.legend(fontsize=8)
    axes.flat[-1].axis("off")
    variance = exploration["pca_variance_ratio"]
    axes.flat[-1].text(0.04, 0.75,
        f"Exploratory projection\n\nPC1: {variance[0]:.1%} variance\nPC2: {variance[1]:.1%} variance\n\n"
        "Cluster IDs are arbitrary.\nDBSCAN -1 denotes noise.\nModels use all 30 scaled features;\nPCA is only used for this plot.",
        transform=axes.flat[-1].transAxes, va="top", fontsize=11)
    fig.savefig(output / "clustering_pca.png", dpi=160)
    plt.close(fig)


def write_results(results, exploration, metadata, output):
    output.mkdir(parents=True, exist_ok=True)
    for name in ["fold_metrics", "summary", "predictions", "paired_differences", "feature_importance"]:
        save_csv(results[name], output / f"{name}.csv")
    for name in ["searches", "splits"]:
        (output / f"{name}.json").write_text(json.dumps(results[name], indent=2), encoding="utf-8")
    for name in ["eda", "correlations", "clustering", "assignments"]:
        save_csv(exploration[name], output / f"{name}.csv", index=name in ["eda", "correlations"])
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    save_plots(results, exploration, output)

    summary = results["summary"].sort_values("roc_auc_mean", ascending=False)
    table = summary[["model"]].copy()
    for metric in ["accuracy", "roc_auc", "f1_malignant", "recall_malignant"]:
        table[metric] = [f"{mean:.4f} ± {sd:.4f}" for mean, sd in
                         zip(summary[f"{metric}_mean"], summary[f"{metric}_std"])]
    paired = results["paired_differences"]
    differences = "\n".join(
        f"- {metric}: mean difference {paired['delta_' + metric].mean():+.4f}; "
        f"voting was higher in {(paired['delta_' + metric] > 1e-12).sum()}/{len(paired)} folds."
        for metric in ["accuracy", "roc_auc", "f1_malignant", "recall_malignant"])
    clustering = exploration["clustering"].round(4).astype(object)
    clustering = clustering.where(clustering.notna(), "N/A")
    importance = results["feature_importance"]
    if len(importance):
        importance = (importance.groupby("feature")[["impurity_importance", "permutation_auc_decrease"]]
                      .mean().sort_values("permutation_auc_decrease", ascending=False).reset_index())
        save_csv(importance, output / "feature_importance_summary.csv")
        importance_text = markdown_table(importance.head(10).round(5))
    else:
        importance_text = "No random-forest importance was computed."
    title = "Smoke-test output — not a benchmark" if metadata["quick"] else "Nested cross-validation results"
    report = f"""# {title}

This run uses {metadata['samples']} WDBC observations and {metadata['features']} features.
The positive class is malignant (1); benign is 0. Classification uses a fixed
probability threshold of 0.5. The outer and inner fold counts are
{metadata['outer_folds']} and {metadata['inner_folds']}; the random seed is {metadata['seed']}.

## Classification

Values are the unweighted means and sample standard deviations across outer folds.
They are not training scores or independent external-validation estimates.

{markdown_table(table)}

`selected_single` selects one of the five model families in each outer fold using
only inner-CV ROC AUC. It is a model-selection procedure, not one fixed model.
`weighted_voting` selects weights for fixed logistic-regression, SVM and random-forest
members using inner CV. The candidate grids are saved in `searches.json`.

![Classification comparison](model_comparison.png)

## Does probability-weighted voting help?

The paired differences below are voting minus `selected_single` on the same outer folds.

{differences}

These are descriptive differences. Five overlapping CV training sets do not provide
five independent experiments. No statistical-significance or clinical-benefit claim
is made. The ensemble is retained even if it underperforms.

![Out-of-fold ROC curves](roc_curves.png)

ROC curves pool cross-fitted probabilities for visualization; the table averages
fold-level AUC values, which need not equal the pooled AUC.

## Random-forest feature analysis

The table averages importances across outer folds. Permutation importance measures
the decrease in outer-fold AUC when a feature is shuffled. It is computed after
fitting and is not used for feature selection. Correlated predictors can share or
mask importance; these rankings do not establish causal or clinical importance.

{importance_text}

## Exploratory clustering

These analyses use all samples and are not out-of-sample classification results.
Features are standardized before clustering; PCA is a display projection only.
The IQR analysis counts unusual values and does not delete or replace observations.
DBSCAN metrics exclude noise observations and must be read together with coverage;
they are not directly comparable to full-coverage metrics. Diagnosis labels are
used only for post-hoc comparison with cluster assignments, not fitting.
N/A means that fewer than two non-noise clusters were available for these metrics.

{markdown_table(clustering)}

![PCA views of clustering](clustering_pca.png)

## Reproduction and interpretation

- `metadata.json`: software versions, dataset hash, code hashes and run settings.
- `splits.json`: outer and inner sample indices; inner indices use global row numbers.
- `searches.json`: selected hyperparameters, candidate scores and selected single-model families.
- `predictions.csv`: one outer-test probability per sample and evaluated method.
- CSVs preserve 17 significant digits. When reloading with pandas, use
  `pd.read_csv(path, float_precision="round_trip")` to preserve near-equal
  probabilities and their ordering for AUC / average-precision calculations.
- `fold_metrics.csv` and `paired_differences.csv`: per-fold values for direct comparison.
- This is a small, previously studied public dataset. Nested CV repairs the evaluation
  procedure but does not create a new external cohort or remove earlier familiarity
  with the dataset. Results support a methodological course project, not a deployed
  diagnostic system.
"""
    (output / "report.md").write_text(report, encoding="utf-8")
