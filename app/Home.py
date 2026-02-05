from __future__ import annotations

from pathlib import Path

import streamlit as st

from core.session import KEYS, ensure_defaults, reset_all


def _load_css() -> None:
    css_path = Path(__file__).parent / "assets" / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _kpi_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="cl-kpi">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _step_card(title: str, desc: str) -> None:
    st.markdown(
        f"""
        <div class="cl-step">
          <div class="title">{title}</div>
          <p class="desc">{desc}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _nav_buttons() -> None:
    left, mid, right = st.columns(3, gap="small")
    with left:
        if st.button("Open Data", type="primary", width="stretch"):
            st.switch_page("pages/1_Data.py")
    with mid:
        if st.button("Open Model", width="stretch"):
            st.switch_page("pages/2_Model.py")
    with right:
        if st.button("Open Results", width="stretch"):
            st.switch_page("pages/3_Results.py")


def main() -> None:
    st.set_page_config(page_title="ClusterLens", page_icon="🔎", layout="wide")
    ensure_defaults()
    _load_css()

    with st.sidebar:
        st.markdown("### ClusterLens")
        st.caption("Unsupervised ML: clustering + anomaly detection")
        if st.button("Reset app", width="stretch"):
            reset_all(keep_dataset=False)
            st.rerun()

        st.divider()
        st.caption("Tips")
        st.write("- Start in **Data** to upload or use a sample CSV.")
        st.write("- Train in **Model**, explore in **Results**.")

    st.markdown(
        """
        <div class="cl-hero">
          <h1>ClusterLens</h1>
          <p class="cl-subtle">Upload a CSV, discover clusters, and detect anomalies with an end-to-end unsupervised ML workflow.</p>
          <span class="cl-pill">Clustering</span>
          <span class="cl-pill">Anomaly detection</span>
          <span class="cl-pill">PCA visualization</span>
          <span class="cl-pill">Export results</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    _nav_buttons()

    st.write("")
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        st.markdown("### Workflow")
        steps = st.columns(3, gap="small")
        with steps[0]:
            _step_card("1) Data", "Upload a CSV or load the sample dataset. Pick feature columns (no labels needed).")
        with steps[1]:
            _step_card("2) Model", "Choose clustering or anomaly detection, tune params, and train a pipeline.")
        with steps[2]:
            _step_card("3) Results", "Explore PCA plots, inspect clusters/outliers, and download results + model artifact.")

        st.write("")
        st.markdown("### What this app supports")
        st.markdown(
            """
            - Mixed **numeric + categorical** features (impute, one-hot, scale)
            - Clustering: **K-Means**, **DBSCAN**, **Gaussian Mixture**
            - Anomaly detection: **Isolation Forest**, **Local Outlier Factor**
            - Fast 2D visualization via **PCA**
            - Export: predictions CSV + portable **model artifact**
            """
        )

    with right:
        st.markdown("### Project status")
        df = st.session_state[KEYS.dataframe]
        pipeline = st.session_state[KEYS.pipeline]
        result = st.session_state[KEYS.results]

        k1, k2, k3 = st.columns(3, gap="small")
        with k1:
            _kpi_card("Dataset", "Loaded" if df is not None else "Not loaded")
        with k2:
            _kpi_card("Model", "Trained" if pipeline is not None else "Not trained")
        with k3:
            _kpi_card("Task", getattr(result, "task", "—") if result is not None else "—")

        st.markdown('<div class="cl-divider"></div>', unsafe_allow_html=True)

        st.markdown('<div class="cl-card">', unsafe_allow_html=True)
        st.markdown("**Recommended dataset shape**")
        st.write("- Rows: 100+ (more is better)")
        st.write("- Features: 2–50")
        st.write("- Avoid ID-only columns")
        st.markdown("</div>", unsafe_allow_html=True)

        if df is None:
            st.info("Load a dataset in **Data** to begin.")
        else:
            st.success(
                f"Loaded: **{st.session_state[KEYS.dataset_name]}** · {len(df):,} rows · {df.shape[1]} columns"
            )


if __name__ == "__main__":
    main()
