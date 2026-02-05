from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from .preprocess import PreprocessConfig, build_preprocessor


TaskName = Literal["Clustering", "Anomaly detection"]
ClusterModelName = Literal["K-Means", "DBSCAN", "Gaussian Mixture"]
AnomalyModelName = Literal["Isolation Forest", "Local Outlier Factor"]


@dataclass(frozen=True)
class TrainResult:
    task: TaskName
    model_name: str
    used_columns: list[str]
    labels: np.ndarray
    scores: np.ndarray | None
    params: dict[str, Any]


def _build_model(task: TaskName, model_name: str, params: dict[str, Any]) -> BaseEstimator:
    if task == "Clustering":
        if model_name == "K-Means":
            return KMeans(
                n_clusters=int(params.get("n_clusters", 4)),
                n_init="auto",
                random_state=int(params.get("random_state", 42)),
            )
        if model_name == "DBSCAN":
            return DBSCAN(
                eps=float(params.get("eps", 0.5)),
                min_samples=int(params.get("min_samples", 10)),
            )
        if model_name == "Gaussian Mixture":
            return GaussianMixture(
                n_components=int(params.get("n_components", 4)),
                covariance_type=str(params.get("covariance_type", "full")),
                random_state=int(params.get("random_state", 42)),
            )
        raise ValueError("Unknown clustering model.")

    if task == "Anomaly detection":
        if model_name == "Isolation Forest":
            return IsolationForest(
                n_estimators=int(params.get("n_estimators", 300)),
                contamination=float(params.get("contamination", 0.05)),
                random_state=int(params.get("random_state", 42)),
            )
        if model_name == "Local Outlier Factor":
            # For inference later, LOF needs novelty=True (then it supports predict on new data).
            return LocalOutlierFactor(
                n_neighbors=int(params.get("n_neighbors", 35)),
                contamination=float(params.get("contamination", 0.05)),
                novelty=True,
            )
        raise ValueError("Unknown anomaly model.")

    raise ValueError("Unknown task.")


def train(
    *,
    df: pd.DataFrame,
    feature_cols: list[str],
    task: TaskName,
    model_name: str,
    model_params: dict[str, Any],
    preprocess_cfg: PreprocessConfig,
) -> tuple[Pipeline, TrainResult]:
    pre, used_cols = build_preprocessor(df, feature_cols, preprocess_cfg)
    model = _build_model(task, model_name, model_params)

    pipe = Pipeline(steps=[("preprocess", pre), ("model", model)])

    x = df[used_cols]

    if task == "Clustering":
        if hasattr(model, "fit_predict"):
            labels = pipe.fit_predict(x)
        else:
            pipe.fit(x)
            if hasattr(model, "predict"):
                labels = pipe.predict(x)
            else:
                raise ValueError("This clustering model doesn't support prediction.")
        return pipe, TrainResult(
            task=task,
            model_name=model_name,
            used_columns=used_cols,
            labels=np.asarray(labels),
            scores=None,
            params=model_params,
        )

    # Anomaly detection
    pipe.fit(x)
    model_step = pipe.named_steps["model"]
    labels = getattr(model_step, "predict")(pipe.named_steps["preprocess"].transform(x))
    labels = np.asarray(labels)

    scores: np.ndarray | None = None
    if hasattr(model_step, "decision_function"):
        scores = np.asarray(model_step.decision_function(pipe.named_steps["preprocess"].transform(x)))
    elif hasattr(model_step, "score_samples"):
        scores = np.asarray(model_step.score_samples(pipe.named_steps["preprocess"].transform(x)))

    return pipe, TrainResult(
        task=task,
        model_name=model_name,
        used_columns=used_cols,
        labels=labels,
        scores=scores,
        params=model_params,
    )


def predict_with_pipeline(pipe: Pipeline, df: pd.DataFrame, used_cols: list[str], task: TaskName) -> tuple[np.ndarray, np.ndarray | None]:
    x = df[used_cols]
    if task == "Clustering":
        model = pipe.named_steps["model"]
        if hasattr(model, "predict"):
            labels = pipe.predict(x)
            return np.asarray(labels), None
        # Some clustering algorithms (e.g., DBSCAN) do not support predict on new data.
        if hasattr(model, "fit_predict"):
            pre = pipe.named_steps["preprocess"]
            xt = pre.transform(x)
            labels = clone(model).fit_predict(xt)
            return np.asarray(labels), None
        raise ValueError("This clustering model does not support prediction.")

    pre = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]
    xt = pre.transform(x)
    labels = np.asarray(model.predict(xt))
    scores: np.ndarray | None = None
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(xt))
    elif hasattr(model, "score_samples"):
        scores = np.asarray(model.score_samples(xt))
    return labels, scores
