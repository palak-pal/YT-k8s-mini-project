import pandas as pd

from app.core.modeling import predict_with_pipeline, train
from app.core.preprocess import PreprocessConfig


def test_train_clustering_kmeans():
    df = pd.DataFrame({"x": [0, 0, 10, 10], "y": [0, 1, 10, 11], "region": ["A", "A", "B", "B"]})
    pipe, res = train(
        df=df,
        feature_cols=["x", "y", "region"],
        task="Clustering",
        model_name="K-Means",
        model_params={"n_clusters": 2, "random_state": 0},
        preprocess_cfg=PreprocessConfig(),
    )
    assert res.labels.shape[0] == 4
    labels, _ = predict_with_pipeline(pipe, df, res.used_columns, "Clustering")
    assert labels.shape[0] == 4


def test_train_anomaly_isolation_forest():
    df = pd.DataFrame({"x": [0, 0, 0, 100], "y": [0, 1, 2, 100]})
    pipe, res = train(
        df=df,
        feature_cols=["x", "y"],
        task="Anomaly detection",
        model_name="Isolation Forest",
        model_params={"contamination": 0.25, "n_estimators": 50, "random_state": 0},
        preprocess_cfg=PreprocessConfig(),
    )
    assert res.labels.shape[0] == 4
    labels, scores = predict_with_pipeline(pipe, df, res.used_columns, "Anomaly detection")
    assert labels.shape[0] == 4
    assert scores is not None


def test_train_clustering_gaussian_mixture():
    df = pd.DataFrame({"x": [0, 0, 10, 10], "y": [0, 1, 10, 11]})
    pipe, res = train(
        df=df,
        feature_cols=["x", "y"],
        task="Clustering",
        model_name="Gaussian Mixture",
        model_params={"n_components": 2, "covariance_type": "full", "random_state": 0},
        preprocess_cfg=PreprocessConfig(),
    )
    labels, _ = predict_with_pipeline(pipe, df, res.used_columns, "Clustering")
    assert labels.shape[0] == 4
