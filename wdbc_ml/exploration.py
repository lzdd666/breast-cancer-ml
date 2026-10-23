"""Exploratory outputs are separate from predictive evaluation."""

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def cluster_metrics(X, labels, y):
    # DBSCAN's -1 denotes noise, not a biological cluster.
    keep = np.asarray(labels) != -1
    groups = np.unique(np.asarray(labels)[keep])
    result = {"clusters_excluding_noise": len(groups), "coverage": float(keep.mean()),
              "noise_samples": int((~keep).sum()),
              "silhouette": np.nan, "calinski_harabasz": np.nan,
              "davies_bouldin": np.nan, "ari_non_noise": np.nan}
    if 1 < len(groups) < int(keep.sum()):
        values, retained = X[keep], np.asarray(labels)[keep]
        result.update(silhouette=silhouette_score(values, retained),
                      calinski_harabasz=calinski_harabasz_score(values, retained),
                      davies_bouldin=davies_bouldin_score(values, retained),
                      ari_non_noise=adjusted_rand_score(np.asarray(y)[keep], retained))
    return result


def explore(X, y, seed=42):
    lower, upper = X.quantile(0.25), X.quantile(0.75)
    iqr = upper - lower
    outliers = ((X < lower - 1.5 * iqr) | (X > upper + 1.5 * iqr)).sum()
    eda = X.describe().T
    eda["iqr_outlier_count"] = outliers
    scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, svd_solver="full")
    coordinates = pca.fit_transform(scaled)
    algorithms = {
        "kmeans": KMeans(n_clusters=2, n_init=20, random_state=seed),
        "ward": AgglomerativeClustering(n_clusters=2, linkage="ward"),
        "dbscan": DBSCAN(eps=3, min_samples=5),
        "gaussian_mixture": GaussianMixture(n_components=2, n_init=5, random_state=seed),
    }
    records = []
    assignments = pd.DataFrame({"sample_index": np.arange(len(y)), "malignant": y,
                                "pc1": coordinates[:, 0], "pc2": coordinates[:, 1]})
    for name, algorithm in algorithms.items():
        labels = algorithm.fit_predict(scaled)
        assignments[name] = labels
        records.append({"model": name, **cluster_metrics(scaled, labels, y)})
    return {"eda": eda, "correlations": X.corr(method="spearman"),
            "clustering": pd.DataFrame(records), "assignments": assignments,
            "pca_variance_ratio": pca.explained_variance_ratio_.tolist()}
