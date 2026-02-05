from __future__ import annotations

import numpy as np
from sklearn.metrics import silhouette_score


def safe_silhouette(x_2d: np.ndarray, labels: np.ndarray) -> float | None:
    labels = np.asarray(labels)
    unique = [v for v in np.unique(labels) if v != -1]
    if len(unique) < 2:
        return None
    try:
        return float(silhouette_score(x_2d, labels))
    except Exception:
        return None


def anomaly_rate(labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    # sklearn anomaly labels are often {1=inlier, -1=outlier}
    outliers = int(np.sum(labels == -1))
    return outliers / max(len(labels), 1)

