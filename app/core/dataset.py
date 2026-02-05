from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    description: str
    path: Path | None = None


SAMPLE_DATASETS: dict[str, DatasetInfo] = {
    "Sample customers": DatasetInfo(
        name="Sample customers",
        description="Synthetic customer features for clustering + anomaly detection demos.",
        path=Path(__file__).resolve().parents[2] / "data" / "sample_customers.csv",
    ),
}


@st.cache_data(show_spinner=False)
def read_csv_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(data))


@st.cache_data(show_spinner=False)
def read_csv_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def generate_sample_customers(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Three rough "segments" to make clustering intuitive in the UI.
    segment = rng.choice(["Value", "Balanced", "Premium"], size=n, p=[0.45, 0.4, 0.15])

    age = np.clip(
        rng.normal(loc=np.where(segment == "Premium", 48, np.where(segment == "Value", 28, 36)), scale=9),
        18,
        75,
    ).round().astype(int)

    income = np.clip(
        rng.normal(
            loc=np.where(segment == "Premium", 125_000, np.where(segment == "Value", 48_000, 78_000)),
            scale=np.where(segment == "Premium", 18_000, np.where(segment == "Value", 12_000, 15_000)),
        ),
        18_000,
        250_000,
    ).round().astype(int)

    spend = np.clip(
        rng.normal(
            loc=np.where(segment == "Premium", 24, np.where(segment == "Value", 70, 48)),
            scale=10,
        ),
        1,
        100,
    ).round().astype(int)

    tenure = np.clip(
        rng.normal(loc=np.where(segment == "Premium", 72, np.where(segment == "Value", 10, 30)), scale=18),
        1,
        120,
    ).round().astype(int)

    complaints = np.clip(rng.poisson(lam=np.where(segment == "Value", 0.4, 0.2)), 0, 6).astype(int)

    avg_cart = np.clip(
        rng.normal(
            loc=np.where(segment == "Premium", 105, np.where(segment == "Value", 38, 62)),
            scale=12,
        ),
        5,
        250,
    ).round(2)

    region = rng.choice(["North", "South", "East", "West"], size=n, p=[0.26, 0.25, 0.24, 0.25])
    channel = rng.choice(["Web", "Store", "Partner"], size=n, p=[0.55, 0.35, 0.10])

    df = pd.DataFrame(
        {
            "customer_id": np.arange(10000, 10000 + n, dtype=int),
            "age": age,
            "annual_income": income,
            "spend_score": spend,
            "tenure_months": tenure,
            "region": region,
            "channel": channel,
            "complaints_last_year": complaints,
            "avg_cart_value": avg_cart,
        }
    )

    # Add a bit of missingness for realistic preprocessing.
    for col in ["annual_income", "avg_cart_value", "region"]:
        mask = rng.random(n) < 0.02
        df.loc[mask, col] = np.nan

    return df


def load_dataset(
    *,
    kind: Literal["sample", "upload"],
    sample_name: str | None = None,
    upload_bytes: bytes | None = None,
    upload_filename: str | None = None,
) -> tuple[pd.DataFrame, str]:
    if kind == "sample":
        if not sample_name or sample_name not in SAMPLE_DATASETS:
            raise ValueError("Unknown sample dataset.")
        info = SAMPLE_DATASETS[sample_name]
        df: pd.DataFrame
        if info.path and info.path.exists():
            df = read_csv_path(str(info.path))
            if len(df) < 200:
                # Keep the UI useful even if the on-disk sample is tiny.
                df = generate_sample_customers()
        else:
            df = generate_sample_customers()
        return df, info.name

    if kind == "upload":
        if upload_bytes is None:
            raise ValueError("Upload bytes missing.")
        df = read_csv_bytes(upload_bytes)
        dataset_name = upload_filename or "Uploaded CSV"
        return df, dataset_name

    raise ValueError("Unknown dataset kind.")


def dataframe_profile(df: pd.DataFrame) -> dict[str, object]:
    n_rows, n_cols = df.shape
    missing_total = int(df.isna().sum().sum())
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    return {
        "rows": n_rows,
        "cols": n_cols,
        "missing_total": missing_total,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }
