"""Nested evaluation with auditable splits and paired outer-fold comparisons."""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

METRICS = ["accuracy", "roc_auc", "average_precision", "f1_malignant",
           "recall_malignant", "precision_malignant"]


def metric_values(y, probabilities):
    predictions = (np.asarray(probabilities) >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, predictions),
        "roc_auc": roc_auc_score(y, probabilities),
        "average_precision": average_precision_score(y, probabilities),
        "f1_malignant": f1_score(y, predictions, zero_division=0),
        "recall_malignant": recall_score(y, predictions, zero_division=0),
        "precision_malignant": precision_score(y, predictions, zero_division=0),
    }


def make_splits(X, y, outer_folds=5, inner_folds=3, seed=42):
    if outer_folds < 2 or inner_folds < 2:
        raise ValueError("Both fold counts must be at least two.")
    if pd.Series(y).value_counts().min() < outer_folds:
        raise ValueError("Too few examples per class for the outer folds.")
    outer = StratifiedKFold(outer_folds, shuffle=True, random_state=seed)
    splits = []
    for fold, (train, test) in enumerate(outer.split(X, y), 1):
        if y.iloc[train].value_counts().min() < inner_folds:
            raise ValueError("Too few training examples per class for the inner folds.")
        inner = StratifiedKFold(inner_folds, shuffle=True, random_state=seed + fold)
        inner_indices = list(inner.split(X.iloc[train], y.iloc[train]))
        splits.append((fold, train, test, inner_indices))
    return splits


def summarize(fold_metrics):
    return (fold_metrics.groupby("model", sort=True)[METRICS].agg(["mean", "std"])
            .pipe(lambda frame: frame.set_axis(
                [f"{metric}_{stat}" for metric, stat in frame.columns], axis=1))
            .reset_index())


def paired_comparison(fold_metrics):
    """Compare voting with the single-model family chosen using inner CV only."""
    selected = fold_metrics.query("model == 'selected_single'").set_index("fold")
    voting = fold_metrics.query("model == 'weighted_voting'").set_index("fold")
    difference = voting[METRICS] - selected[METRICS]
    return difference.add_prefix("delta_").reset_index()


def evaluate(X, y, candidates, outer_folds=5, inner_folds=3, seed=42,
             jobs=1, permutation_repeats=5, progress=print):
    records, predictions, search_log, split_log, importances = [], [], [], [], []
    for fold, train, test, inner_indices in make_splits(
            X, y, outer_folds, inner_folds, seed):
        split_log.append({
            "fold": fold, "train": train.tolist(), "test": test.tolist(),
            "inner": [{"train": train[a].tolist(), "validation": train[b].tolist()}
                      for a, b in inner_indices],
        })
        fitted, inner_scores, probabilities = {}, {}, {}
        for name, (estimator, grid) in candidates.items():
            progress(f"Outer fold {fold}/{outer_folds}: {name}")
            search = GridSearchCV(estimator, grid, scoring="roc_auc", cv=inner_indices,
                                  refit=True, n_jobs=jobs, error_score="raise")
            search.fit(X.iloc[train], y.iloc[train])
            fitted[name] = search.best_estimator_
            inner_scores[name] = float(search.best_score_)
            positive_index = list(search.classes_).index(1)
            probabilities[name] = search.predict_proba(X.iloc[test])[:, positive_index]
            search_log.append({
                "fold": fold, "model": name, "best_parameters": search.best_params_,
                "best_inner_auc": float(search.best_score_),
                "candidates": [{"parameters": params, "mean_inner_auc": float(score)}
                               for params, score in zip(search.cv_results_["params"],
                                                        search.cv_results_["mean_test_score"])],
            })
        # This comparator never chooses a model using outer-test outcomes.
        single_names = [name for name in candidates if name != "weighted_voting"]
        winner = max(single_names, key=inner_scores.get)
        probabilities["selected_single"] = probabilities[winner].copy()
        search_log.append({"fold": fold, "model": "selected_single",
                           "selected_family": winner, "best_inner_auc": inner_scores[winner]})
        for name, probability in probabilities.items():
            records.append({"fold": fold, "model": name,
                            **metric_values(y.iloc[test], probability)})
            predictions.extend({"sample_index": int(index), "fold": fold, "model": name,
                                "y_true": int(label), "p_malignant": float(p),
                                "y_pred": int(p >= 0.5)}
                               for index, label, p in zip(test, y.iloc[test], probability))
        if "random_forest" in fitted:
            rf = fitted["random_forest"]
            impurity = rf.named_steps["model"].feature_importances_
            if permutation_repeats:
                # Interpret outer-test predictions only after fitting. These scores
                # are never fed back into model tuning or feature selection.
                perm = permutation_importance(rf, X.iloc[test], y.iloc[test],
                    scoring="roc_auc", n_repeats=permutation_repeats,
                    random_state=seed + fold, n_jobs=jobs)
                permutation = perm.importances_mean
            else:
                permutation = np.full(X.shape[1], np.nan)
            importances.extend({"fold": fold, "feature": feature,
                                "impurity_importance": float(imp),
                                "permutation_auc_decrease": float(perm)}
                               for feature, imp, perm in zip(X.columns, impurity, permutation))
    frame = pd.DataFrame(records)
    return {"fold_metrics": frame, "summary": summarize(frame),
            "predictions": pd.DataFrame(predictions), "searches": search_log,
            "splits": split_log, "paired_differences": paired_comparison(frame),
            "feature_importance": pd.DataFrame(importances)}
