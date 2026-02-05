from __future__ import annotations

import streamlit as st

from core.modeling import (
    AnomalyModelName,
    ClusterModelName,
    TaskName,
    train,
)
from core.preprocess import PreprocessConfig, safe_row_sample
from core.session import KEYS, ensure_defaults


def _clustering_params_ui(model_name: ClusterModelName) -> dict:
    if model_name == "K-Means":
        return {
            "n_clusters": st.slider("Number of clusters", 2, 20, 4),
            "random_state": st.number_input("Random seed", 0, 10_000, 42),
        }
    if model_name == "DBSCAN":
        return {
            "eps": st.slider("eps", 0.05, 5.0, 0.5),
            "min_samples": st.slider("min_samples", 2, 100, 10),
        }
    # Gaussian Mixture
    return {
        "n_components": st.slider("Components", 2, 20, 4),
        "covariance_type": st.selectbox("Covariance type", ["full", "tied", "diag", "spherical"], index=0),
        "random_state": st.number_input("Random seed", 0, 10_000, 42),
    }


def _anomaly_params_ui(model_name: AnomalyModelName) -> dict:
    if model_name == "Isolation Forest":
        return {
            "contamination": st.slider("Contamination (expected outlier rate)", 0.001, 0.3, 0.05, format="%.3f"),
            "n_estimators": st.slider("Trees", 50, 800, 300, step=50),
            "random_state": st.number_input("Random seed", 0, 10_000, 42),
        }
    return {
        "contamination": st.slider("Contamination (expected outlier rate)", 0.001, 0.3, 0.05, format="%.3f"),
        "n_neighbors": st.slider("Neighbors", 5, 200, 35),
    }


def _preprocess_ui() -> PreprocessConfig:
    with st.expander("Preprocessing", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            numeric_impute = st.selectbox("Numeric impute", ["median", "mean"], index=0)
            scale_numeric = st.toggle("Scale numeric features", value=True)
        with col2:
            categorical_impute = st.selectbox("Categorical impute", ["most_frequent"], index=0)
            one_hot = st.toggle("One-hot encode categoricals", value=True)
        with col3:
            drop_high = st.toggle("Drop high-cardinality categoricals", value=True, help="Helps avoid huge one-hot matrices.")
            max_cat = st.slider("Max categories", 5, 200, 50, disabled=not drop_high)

    return PreprocessConfig(
        numeric_impute=numeric_impute,
        categorical_impute=categorical_impute,
        scale_numeric=scale_numeric,
        one_hot=one_hot,
        drop_high_cardinality=drop_high,
        max_categories=max_cat,
    )


def main() -> None:
    ensure_defaults()
    st.title("Model")
    st.caption("Pick an unsupervised task, tune parameters, and train the model.")

    df = st.session_state[KEYS.dataframe]
    if df is None:
        st.info("Load a dataset first in the **Data** page.")
        return

    feature_cols = st.session_state[KEYS.feature_cols]
    if not feature_cols:
        st.warning("Pick feature columns in the **Data** page before training.")
        return

    task: TaskName = st.radio("Task", ["Clustering", "Anomaly detection"], horizontal=True)
    st.session_state[KEYS.task] = task

    if task == "Clustering":
        model_name: ClusterModelName = st.selectbox("Algorithm", ["K-Means", "DBSCAN", "Gaussian Mixture"])
        params = _clustering_params_ui(model_name)
    else:
        model_name: AnomalyModelName = st.selectbox("Algorithm", ["Isolation Forest", "Local Outlier Factor"])
        params = _anomaly_params_ui(model_name)

    st.session_state[KEYS.model_name] = model_name
    st.session_state[KEYS.params] = params
    preprocess_cfg = _preprocess_ui()

    st.divider()
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.markdown("#### Training")
        st.write(f"Dataset: **{st.session_state[KEYS.dataset_name]}**")
        st.write(f"Features: **{len(feature_cols)}** selected")

        if len(df) < 10:
            st.warning("Dataset is very small; results may be unstable.")
        min_rows = min(200, len(df))
        max_rows = min(50_000, len(df))
        default_rows = min(5000, len(df))
        train_rows = st.slider("Rows to use (sampled)", min_rows, max_rows, default_rows)
        train_df = safe_row_sample(df, max_rows=train_rows)

        if st.button("Train model", type="primary", width="stretch"):
            st.session_state[KEYS.last_error] = None
            with st.spinner("Training…"):
                try:
                    pipe, res = train(
                        df=train_df,
                        feature_cols=feature_cols,
                        task=task,
                        model_name=model_name,
                        model_params=params,
                        preprocess_cfg=preprocess_cfg,
                    )
                    st.session_state[KEYS.pipeline] = pipe
                    st.session_state[KEYS.results] = res
                    st.success("Training complete. Open the **Results** page.")
                except Exception as e:
                    st.session_state[KEYS.last_error] = str(e)
                    st.error(f"Training failed: {e}")

    with right:
        st.markdown("#### Notes")
        st.write("- Use **K-Means** for compact spherical clusters.")
        st.write("- Use **DBSCAN** for arbitrary shapes + noise handling.")
        st.write("- Use **Isolation Forest** for strong baseline outlier detection.")
        st.write("- If results look messy, try scaling + fewer features.")
        if st.session_state.get(KEYS.last_error):
            st.divider()
            st.code(st.session_state[KEYS.last_error], language="text")


if __name__ == "__main__":
    main()
