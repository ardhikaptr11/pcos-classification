import re
from typing import Any, Literal, Union

import numpy as np
import pandas as pd

from .schemas import FeatureMap


def convert_to_numeric(
    df: pd.DataFrame, errors: Literal["coerce", "raise"] = "raise"
) -> pd.DataFrame:
    df_result = df.copy()

    features = df_result.select_dtypes(include="object").columns.tolist()

    for feature in features:
        df_result[feature] = pd.to_numeric(
            df_result[feature], errors=errors
        )  # Coerce will result in NaN if any value cannot be converted

    return df_result


def cast_type(
    df: pd.DataFrame, cols: Union[str, list[str]], target_type: Any
) -> pd.DataFrame:
    df_result = df.copy()

    if not isinstance(cols, (str, list)):
        return df_result

    if isinstance(cols, list):
        for col in cols:
            if col in df_result.columns:
                df_result[col] = df_result[col].astype(dtype=target_type)
    else:
        if cols in df_result.columns:
            df_result[cols] = df_result[cols].astype(dtype=target_type)

    return df_result


def create_new_features(df: pd.DataFrame, fmap: list[FeatureMap]) -> pd.DataFrame:
    df_result = df.copy()

    for item in fmap:
        name = item.get("name")
        features = item.get("features", [])
        how = item.get("how")

        if not name or name in df_result.columns:
            continue

        if callable(how):
            df_result[name] = how(df_result)
        elif isinstance(features, list) and len(features) >= 2:
            if not all(f in df_result.columns for f in features):
                continue

            if how == "add":
                df_result[name] = df_result[features].sum(axis=1)
            elif how == "sub":
                result = df_result[features[0]].copy()
                for f in features[1:]:
                    result = abs(result - df_result[f])
                df_result[name] = result
            elif how == "mul":
                df_result[name] = df_result[features].prod(axis=1)
            elif how == "div":
                result = df_result[features[0]].copy()
                for f in features[1:]:
                    result = result / df_result[f].replace(
                        0, np.nan
                    )  # Safety guard to avoid ZeroDivisionError
                df_result[name] = result

    return df_result


def standardize_col_names(df: pd.DataFrame) -> pd.DataFrame:
    df_result = df.copy()

    new_col_names = []
    for col in df_result.columns:
        col = str(col).strip()
        col = re.sub(r"[^\w\s]", "", col)  # Delete special characters
        col = re.sub(
            r"\s+", "_", col
        )  # Replace spaces (single/double) with underscores
        col = re.sub(
            r"_+", "_", col
        )  # Fix double or more underscore if detected after converting
        new_col_names.append(col.lower())

    df_result.columns = new_col_names

    return df_result


def repos_col(df: pd.DataFrame) -> pd.DataFrame:
    df_result = df.copy()
    cols = df_result.columns.tolist()

    def move_before(col: str, ref_col: str):
        if col in cols and ref_col in cols:
            cols.remove(col)
            ref_idx = cols.index(ref_col)
            cols.insert(ref_idx, col)

    def move_after(col: str, ref_col: str):
        if col in cols and ref_col in cols:
            cols.remove(col)
            ref_idx = cols.index(ref_col) + 1
            cols.insert(ref_idx, col)

    target_last = "PCOS (Y/N)"
    if target_last in cols:
        cols.remove(target_last)
        cols.append(target_last)

    move_before(col="LH/FSH Ratio", ref_col="Hip (inch)")
    move_before(col="Total Follicles", ref_col="Avg. F size (L) (mm)")
    move_after(col="Follicles Difference", ref_col="Total Follicles")

    return df_result[cols]
