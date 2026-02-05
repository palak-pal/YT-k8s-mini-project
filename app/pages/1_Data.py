from __future__ import annotations

import pandas as pd
import streamlit as st

from core.dataset import SAMPLE_DATASETS, dataframe_profile, load_dataset
from core.preprocess import infer_feature_columns, safe_row_sample
from core.session import KEYS, ensure_defaults, set_dataframe


def _render_overview(df: pd.DataFrame) -> None:
    prof = dataframe_profile(df)

    a, b, c, d = st.columns(4)
    a.metric("Rows", f"{prof['rows']:,}")
    b.metric("Columns", f"{prof['cols']:,}")
    c.metric("Numeric cols", f"{len(prof['numeric_cols']):,}")
    d.metric("Missing values", f"{prof['missing_total']:,}")

    with st.expander("Preview", expanded=True):
        st.dataframe(df.head(50), width="stretch")

    with st.expander("Column types", expanded=False):
        st.write("**Numeric**")
        st.write(prof["numeric_cols"] or "—")
        st.write("**Categorical / other**")
        st.write(prof["categorical_cols"] or "—")


def main() -> None:
    ensure_defaults()
    st.title("Data")
    st.caption("Load a CSV and pick the feature columns you want the model to use.")

    tab1, tab2 = st.tabs(["Sample dataset", "Upload CSV"])

    with tab1:
        sample_name = st.selectbox("Choose a sample dataset", options=list(SAMPLE_DATASETS.keys()))
        st.caption(SAMPLE_DATASETS[sample_name].description)
        if st.button("Load sample", type="primary"):
            df, name = load_dataset(kind="sample", sample_name=sample_name)
            set_dataframe(df, name)
            st.success(f"Loaded: {name}")
            st.rerun()

    with tab2:
        uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded is not None:
            if st.button("Load upload", type="primary"):
                df, name = load_dataset(
                    kind="upload",
                    upload_bytes=uploaded.getvalue(),
                    upload_filename=uploaded.name,
                )
                set_dataframe(df, name)
                st.success(f"Loaded: {name}")
                st.rerun()

    df = st.session_state[KEYS.dataframe]
    if df is None:
        st.info("Load a dataset to continue.")
        return

    st.divider()
    st.subheader("Dataset overview")
    _render_overview(df)

    st.subheader("Feature selection")
    default_cols = st.session_state[KEYS.feature_cols] or infer_feature_columns(df)
    feature_cols = st.multiselect(
        "Columns used for training",
        options=list(df.columns),
        default=default_cols,
        help="Pick the columns used by the model. Targets/labels are not needed for unsupervised ML.",
    )

    st.session_state[KEYS.feature_cols] = feature_cols
    st.caption("Large datasets are sampled for visualization/training to keep the app snappy.")

    sampled = safe_row_sample(df, max_rows=5000)
    with st.expander("Training sample (up to 5,000 rows)", expanded=False):
        st.dataframe(sampled.head(50), width="stretch")


if __name__ == "__main__":
    main()
