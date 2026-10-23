"""Prespecified model families and small inner-CV search spaces."""

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def pipeline(classifier, scale=True):
    return Pipeline([
        ("scale", StandardScaler() if scale else "passthrough"),
        ("model", classifier),
    ])


def make_candidates(seed=42, quick=False):
    """Every estimator is fresh; scaling is fitted inside each CV training fold.

    The voting experiment uses fixed base-model settings and selects only its
    weights. Its base models are not the separately tuned single-model winners.
    """
    trees = 30 if quick else 200
    candidates = {
        "naive_bayes": (pipeline(GaussianNB()),
                        {"model__var_smoothing": [1e-9, 1e-8, 1e-7]}),
        "logistic_regression": (
            pipeline(LogisticRegression(max_iter=5000, random_state=seed)),
            {"model__C": [0.1, 1.0, 10.0]}),
        "svm": (pipeline(SVC(probability=True, random_state=seed)),
                {"model__C": [0.1, 1.0, 10.0]}),
        "decision_tree": (
            pipeline(DecisionTreeClassifier(random_state=seed), scale=False),
            {"model__max_depth": [3, 5, None], "model__min_samples_leaf": [1, 3]}),
        "random_forest": (
            pipeline(RandomForestClassifier(n_estimators=trees, random_state=seed,
                                            n_jobs=1), scale=False),
            {"model__min_samples_leaf": [1, 3], "model__max_features": ["sqrt", 0.5]}),
    }
    voter = VotingClassifier(estimators=[
        ("lr", pipeline(LogisticRegression(C=1.0, max_iter=5000, random_state=seed))),
        ("svm", pipeline(SVC(C=1.0, probability=True, random_state=seed))),
        ("rf", pipeline(RandomForestClassifier(n_estimators=trees,
                                              random_state=seed, n_jobs=1), scale=False)),
    ], voting="soft", n_jobs=1)
    candidates["weighted_voting"] = (
        voter, {"weights": [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2)]})
    if quick:
        candidates = {name: (est, {key: values[:1] for key, values in grid.items()})
                      for name, (est, grid) in candidates.items()}
    return candidates
