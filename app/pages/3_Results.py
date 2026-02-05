from __future__ import annotations

import pandas as pd
import streamlit as st

from core.artifacts import ArtifactMeta, build_artifact_bytes
from core.metrics import anomaly_rate, safe_silhouette
from core.modeling import TrainResult, predict_with_pipeline
from core.preprocess import safe_row_sample
from core.session import KEYS, ensure_defaults
from core.viz import pca_project, plotly_scatter


def _attach_results(df: pd.DataFrame, res: TrainResult) -> pd.DataFrame:
    out = df.copy()
    out["label"] = res.labels
    if res.scores is not None:
        out["score"] = res.scores
    if res.task == "Anomaly detection":
        out["is_outlier"] = out["label"] == -1
    return out


def main() -> None:
    ensure_defaults()
    st.title("Results")
    st.caption("Explore clusters/outliers, visualize in 2D, and export results + model artifact.")

    df = st.session_state[KEYS.dataframe]
    pipe = st.session_state[KEYS.pipeline]
    res: TrainResult | None = st.session_state[KEYS.results]

    if df is None:
        st.info("Load a dataset in **Data**.")
        return

    if pipe is None or res is None:
        st.info("Train a model in **Model** first.")
        return

    st.subheader("Summary")
    a, b, c = st.columns(3)
    a.metric("Task", res.task)
    b.metric("Model", res.model_name)
    c.metric("Used columns", str(len(res.used_columns)))
    if res.task == "Clustering":
        model = pipe.named_steps.get("model")
        if model is not None and (not hasattr(model, "predict")) and hasattr(model, "fit_predict"):
            st.warning("Note: this clustering algorithm doesn't support `predict()`. For plots/exports, labels are computed by refitting on the selected rows.")

    st.divider()
    st.subheader("Visualize (PCA 2D)")

    min_plot = min(200, len(df))
    max_plot = min(20_000, len(df))
    default_plot = min(5000, len(df))
    viz_rows = st.slider("Rows to plot", min_plot, max_plot, default_plot)
    viz_df = safe_row_sample(df, max_rows=viz_rows)

    with st.spinner("Computing projection…"):
        proj = pca_project(pipe, viz_df, res.used_columns, max_rows=20_000)

    if res.task == "Clustering":
        labels, _ = predict_with_pipeline(pipe, viz_df, res.used_columns, res.task)
        fig = plotly_scatter(
            base_df=viz_df,
            xy=proj.df,
            color=labels,
            hover_cols=res.used_columns[:8],
            title="Clusters (PCA projection)",
            color_name="cluster",
        )
        st.plotly_chart(fig, width="stretch")

        if proj.explained_variance:
            st.caption(f"PCA explained variance: PC1={proj.explained_variance[0]:.2%}, PC2={proj.explained_variance[1]:.2%}")
        # silhouette computed on PCA space for speed/UX
        sil = safe_silhouette(proj.df[["x", "y"]].to_numpy(), labels)
        if sil is not None:
            st.info(f"Silhouette (on PCA 2D): **{sil:.3f}**")
        else:
            st.info("Silhouette not available (needs 2+ clusters; DBSCAN noise can also block it).")
    else:
        labels, scores = predict_with_pipeline(pipe, viz_df, res.used_columns, res.task)
        color = (labels == -1).astype(int)
        fig = plotly_scatter(
            base_df=viz_df,
            xy=proj.df,
            color=color,
            hover_cols=res.used_columns[:8],
            title="Outliers (PCA projection)",
            color_name="outlier",
        )
        st.plotly_chart(fig, width="stretch")
        st.info(f"Outlier rate (in plotted sample): **{anomaly_rate(labels):.2%}**")
        if scores is not None:
            st.caption("Score meaning depends on the algorithm; lower is often 'more anomalous'.")

    st.divider()
    st.subheader("Export")

    min_export = min(200, len(df))
    export_rows = st.slider("Rows to export", min_export, len(df), min(10_000, len(df)))
    export_df = safe_row_sample(df, max_rows=export_rows)
    export_labels, export_scores = predict_with_pipeline(pipe, export_df, res.used_columns, res.task)
    export_res = TrainResult(
        task=res.task,
        model_name=res.model_name,
        used_columns=res.used_columns,
        labels=export_labels,
        scores=export_scores,
        params=res.params,
    )
    out_df = _attach_results(export_df, export_res)

    st.download_button(
        "Download results CSV",
        data=out_df.to_csv(index=False).encode("utf-8"),
        file_name="clusterlens_results.csv",
        mime="text/csv",
        width="stretch",
    )

    if st.session_state[KEYS.artifact_bytes] is None:
        meta = ArtifactMeta(
            app="ClusterLens",
            task=res.task,
            model_name=res.model_name,
            used_columns=res.used_columns,
            params=res.params,
        )
        st.session_state[KEYS.artifact_bytes] = build_artifact_bytes(pipeline=pipe, meta=meta)

    st.download_button(
        "Download model artifact (.zip)",
        data=st.session_state[KEYS.artifact_bytes],
        file_name="clusterlens_model.zip",
        mime="application/zip",
        width="stretch",
        help="Contains `model.joblib` and `meta.json` for reproducible inference.",
    )

    with st.expander("Export preview", expanded=False):
        st.dataframe(out_df.head(50), width="stretch")


if __name__ == "__main__":
    main()
