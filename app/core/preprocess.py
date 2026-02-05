from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class PreprocessConfig:
    numeric_impute: str = "median"
    categorical_impute: str = "most_frequent"
    scale_numeric: bool = True
    one_hot: bool = True
    drop_high_cardinality: bool = True
    max_categories: int = 50


def infer_feature_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        if not s.notna().any():
            continue
        nunique = int(s.nunique(dropna=True))
        if nunique <= 1:
            continue
        # Heuristics: ID-like columns rarely help unsupervised structure.
        if c.lower().endswith("id") or c.lower() in {"id", "uuid"}:
            if nunique >= max(20, int(0.9 * n)):
                continue
        if nunique >= max(50, int(0.95 * n)) and pd.api.types.is_integer_dtype(s):
            continue
        cols.append(c)
    return cols


def split_columns(df: pd.DataFrame, feature_cols: list[str]) -> tuple[list[str], list[str]]:
    numeric = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in feature_cols if c not in numeric]
    return numeric, categorical


def _filter_high_cardinality(df: pd.DataFrame, categorical_cols: list[str], max_categories: int) -> list[str]:
    kept: list[str] = []
    for c in categorical_cols:
        nunique = int(df[c].nunique(dropna=True))
        if nunique <= max_categories:
            kept.append(c)
    return kept


def build_preprocessor(df: pd.DataFrame, feature_cols: list[str], cfg: PreprocessConfig) -> tuple[ColumnTransformer, list[str]]:
    if not feature_cols:
        raise ValueError("Select at least one feature column.")

    numeric_cols, categorical_cols = split_columns(df, feature_cols)

    if cfg.drop_high_cardinality and categorical_cols:
        categorical_cols = _filter_high_cardinality(df, categorical_cols, cfg.max_categories)

    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_cols:
        steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy=cfg.numeric_impute))]
        if cfg.scale_numeric:
            steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(steps=steps), numeric_cols))

    if categorical_cols:
        steps = [("imputer", SimpleImputer(strategy=cfg.categorical_impute))]
        if cfg.one_hot:
            steps.append(
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                )
            )
        transformers.append(("cat", Pipeline(steps=steps), categorical_cols))

    if not transformers:
        raise ValueError("No usable feature columns after preprocessing filters.")

    pre = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)

    used_cols = numeric_cols + categorical_cols
    return pre, used_cols


def safe_row_sample(df: pd.DataFrame, max_rows: int = 5000, random_state: int = 42) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=random_state)


def to_numpy_2d(x: object) -> np.ndarray:
    if isinstance(x, np.ndarray):
        if x.ndim == 1:
            return x.reshape(-1, 1)
        return x
    return np.asarray(x)
