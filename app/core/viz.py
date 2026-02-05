from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class Projection2D:
    df: pd.DataFrame
    explained_variance: tuple[float, float] | None


def pca_project(pipe: Pipeline, df: pd.DataFrame, used_cols: list[str], max_rows: int = 10000) -> Projection2D:
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)

    pre = pipe.named_steps["preprocess"]
    x = pre.transform(df[used_cols])
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] < 2:
        proj = pd.DataFrame({"x": np.zeros(len(df)), "y": np.zeros(len(df))})
        return Projection2D(df=proj, explained_variance=None)

    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(x)
    proj = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1]})
    var = tuple(float(v) for v in pca.explained_variance_ratio_[:2])
    return Projection2D(df=proj, explained_variance=var)


def plotly_scatter(
    *,
    base_df: pd.DataFrame,
    xy: pd.DataFrame,
    color: np.ndarray,
    hover_cols: list[str],
    title: str,
    color_name: str,
) -> Any:
    import plotly.express as px

    plot_df = base_df.reset_index(drop=True).copy()
    plot_df["x"] = xy["x"].to_numpy()
    plot_df["y"] = xy["y"].to_numpy()
    plot_df[color_name] = color

    hover_data = {c: True for c in hover_cols if c in plot_df.columns}
    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color=color_name,
        hover_data=hover_data,
        title=title,
        height=520,
    )
    fig.update_traces(marker=dict(size=7, opacity=0.85))
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), legend_title_text=color_name)
    return fig

