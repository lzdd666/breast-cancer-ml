"""Load the public dataset with an explicit positive-class convention."""

import hashlib

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_data():
    """Return 30 features and labels with malignant=1, benign=0.

    scikit-learn uses the opposite label convention in the bundled dataset.
    No network request or separately downloaded data file is required.
    """
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data.copy()
    y = pd.Series((dataset.target == 0).astype(int), name="malignant")
    return X, y


def fingerprint(X, y):
    digest = hashlib.sha256()
    digest.update("\n".join(X.columns).encode("utf-8"))
    digest.update(np.asarray(X, dtype="<f8").tobytes())
    digest.update(np.asarray(y, dtype="<i8").tobytes())
    return digest.hexdigest()
