from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SessionKeys:
    dataframe: str = "cl_df"
    dataset_name: str = "cl_dataset_name"
    feature_cols: str = "cl_feature_cols"
    task: str = "cl_task"
    model_name: str = "cl_model_name"
    params: str = "cl_params"
    pipeline: str = "cl_pipeline"
    results: str = "cl_results"
    projection: str = "cl_projection"
    artifact_bytes: str = "cl_artifact_bytes"
    last_error: str = "cl_last_error"


KEYS = SessionKeys()


def ensure_defaults() -> None:
    import streamlit as st

    defaults: dict[str, Any] = {
        KEYS.dataframe: None,
        KEYS.dataset_name: "Sample customers",
        KEYS.feature_cols: [],
        KEYS.task: "Clustering",
        KEYS.model_name: "K-Means",
        KEYS.params: {},
        KEYS.pipeline: None,
        KEYS.results: None,
        KEYS.projection: None,
        KEYS.artifact_bytes: None,
        KEYS.last_error: None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_all(keep_dataset: bool = False) -> None:
    import streamlit as st

    df = st.session_state.get(KEYS.dataframe) if keep_dataset else None
    dataset_name = st.session_state.get(KEYS.dataset_name) if keep_dataset else "Sample customers"
    for key in KEYS.__dict__.values():
        st.session_state.pop(key, None)

    ensure_defaults()
    st.session_state[KEYS.dataframe] = df
    st.session_state[KEYS.dataset_name] = dataset_name


def set_dataframe(df: pd.DataFrame, dataset_name: str) -> None:
    import streamlit as st

    st.session_state[KEYS.dataframe] = df
    st.session_state[KEYS.dataset_name] = dataset_name
    st.session_state[KEYS.results] = None
    st.session_state[KEYS.pipeline] = None
    st.session_state[KEYS.projection] = None
    st.session_state[KEYS.artifact_bytes] = None

