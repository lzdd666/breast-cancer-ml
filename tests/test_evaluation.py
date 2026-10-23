import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from wdbc_ml.data import load_data
from wdbc_ml.evaluation import evaluate, make_splits, metric_values
from wdbc_ml.exploration import cluster_metrics
from wdbc_ml.models import pipeline


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        values, labels = make_classification(n_samples=90, n_features=5,
                                            n_informative=3, random_state=6)
        cls.X = pd.DataFrame(values, columns=list("abcde"))
        cls.y = pd.Series(labels)

    def candidates(self):
        lr = pipeline(LogisticRegression(max_iter=2000, random_state=42))
        tree = pipeline(DecisionTreeClassifier(max_depth=2, random_state=42), scale=False)
        voting = VotingClassifier([("lr", lr), ("tree", tree)], voting="soft")
        return {"logistic_regression": (lr, {"model__C": [0.1, 1.0]}),
                "decision_tree": (tree, {"model__max_depth": [2, 3]}),
                "weighted_voting": (voting, {"weights": [(1, 1), (2, 1)]})}

    def run_evaluation(self, X=None, y=None):
        return evaluate(self.X if X is None else X, self.y if y is None else y,
                        self.candidates(), outer_folds=3, inner_folds=2,
                        permutation_repeats=0, progress=lambda message: None)

    def test_dataset_positive_class(self):
        X, y = load_data()
        self.assertEqual(X.shape, (569, 30))
        self.assertEqual(int(y.sum()), 212)
        self.assertEqual(int((y == 0).sum()), 357)

    def test_nested_splits_exclude_outer_test_samples(self):
        covered = []
        for _, train, test, inner in make_splits(self.X, self.y, 3, 2):
            self.assertFalse(set(train) & set(test))
            covered.extend(test)
            for a, b in inner:
                self.assertFalse(set(train[a]) & set(train[b]))
                self.assertFalse(set(train[a]) & set(test))
                self.assertFalse(set(train[b]) & set(test))
        self.assertEqual(sorted(covered), list(range(len(self.y))))

    def test_scaling_uses_training_data_only(self):
        train, test = self.X.iloc[:60], self.X.iloc[60:] + 10000
        model = pipeline(LogisticRegression(max_iter=2000)).fit(train, self.y.iloc[:60])
        np.testing.assert_allclose(model.named_steps["scale"].mean_, train.mean())
        model.predict_proba(test)
        np.testing.assert_allclose(model.named_steps["scale"].mean_, train.mean())

    def test_outer_labels_do_not_change_inner_selection(self):
        original = self.run_evaluation()
        first_test = original["splits"][0]["test"]
        changed_y = self.y.copy()
        changed_y.iloc[first_test] = 1 - changed_y.iloc[first_test]
        # Keep the same splits, so this directly checks the fitting boundary.
        from unittest.mock import patch
        fixed = make_splits(self.X, self.y, 3, 2)
        with patch("wdbc_ml.evaluation.make_splits", return_value=fixed):
            changed = self.run_evaluation(y=changed_y)
        left = [row for row in original["searches"] if row["fold"] == 1]
        right = [row for row in changed["searches"] if row["fold"] == 1]
        self.assertEqual(left, right)

    def test_predictions_metrics_and_comparator_are_auditable(self):
        result = self.run_evaluation()
        predictions = result["predictions"]
        for name, rows in predictions.groupby("model"):
            self.assertEqual(sorted(rows.sample_index), list(range(len(self.y))))
            self.assertTrue(rows.p_malignant.between(0, 1).all())
            for fold, part in rows.groupby("fold"):
                expected = metric_values(part.y_true, part.p_malignant)
                actual = result["fold_metrics"].query("model == @name and fold == @fold").iloc[0]
                for metric, value in expected.items():
                    self.assertAlmostEqual(actual[metric], value)
        for selection in result["searches"]:
            if selection["model"] == "selected_single":
                fold, family = selection["fold"], selection["selected_family"]
                expected = predictions.query("model == @family and fold == @fold")
                actual = predictions.query("model == 'selected_single' and fold == @fold")
                np.testing.assert_array_equal(expected.p_malignant, actual.p_malignant)

    def test_seeded_evaluation_is_repeatable(self):
        first, second = self.run_evaluation(), self.run_evaluation()
        pd.testing.assert_frame_equal(first["predictions"], second["predictions"])
        self.assertEqual(first["searches"], second["searches"])

    def test_noise_is_not_counted_as_a_cluster(self):
        X = np.arange(12).reshape(6, 2)
        result = cluster_metrics(X, [-1, -1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1])
        self.assertEqual(result["clusters_excluding_noise"], 1)
        self.assertEqual(result["noise_samples"], 2)
        self.assertTrue(np.isnan(result["silhouette"]))

    def test_export_preserves_ranking_of_extreme_probabilities(self):
        from io import StringIO
        from wdbc_ml.reporting import save_csv
        frame = pd.DataFrame({"y_true": [0, 1, 0, 1],
                              "p_malignant": [0.9999999999991, 0.9999999999992,
                                               0.9999999999993, 0.9999999999994]})
        stream = StringIO()
        save_csv(frame, stream)
        stream.seek(0)
        recovered = pd.read_csv(stream, float_precision="round_trip")
        np.testing.assert_array_equal(frame.p_malignant, recovered.p_malignant)
        self.assertEqual(metric_values(frame.y_true, frame.p_malignant),
                         metric_values(recovered.y_true, recovered.p_malignant))


if __name__ == "__main__":
    unittest.main()
