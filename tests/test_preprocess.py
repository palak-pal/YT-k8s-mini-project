import pandas as pd

from app.core.preprocess import PreprocessConfig, build_preprocessor


def test_build_preprocessor_numeric_and_cat():
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, None],
            "b": ["x", "y", None],
            "c": [10, 11, 12],
        }
    )
    pre, used = build_preprocessor(df, ["a", "b", "c"], PreprocessConfig())
    assert set(used) == {"a", "b", "c"}
    xt = pre.fit_transform(df[used])
    assert xt.shape[0] == 3


def test_build_preprocessor_filters_high_cardinality():
    df = pd.DataFrame({"cat": [f"v{i}" for i in range(60)], "num": list(range(60))})
    cfg = PreprocessConfig(drop_high_cardinality=True, max_categories=10)
    pre, used = build_preprocessor(df, ["cat", "num"], cfg)
    assert used == ["num"]
    xt = pre.fit_transform(df[used])
    assert xt.shape == (60, 1)

