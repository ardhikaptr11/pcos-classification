import re
from typing import Any, Callable, Literal, NotRequired, TypedDict, Union

import numpy as np
import pandas as pd


class FeatureMap(TypedDict):
    name: str
    features: NotRequired[list[str]]
    how: Union[str, Callable[[pd.DataFrame], pd.Series]]


def load_dataset(path1: str, path2: str | None = None) -> pd.DataFrame:
    df1 = pd.read_csv(path1)
    df2 = (
        pd.read_excel(path2, sheet_name="Full_new")
        if path2 and path2.endswith(".xlsx")
        else (pd.read_csv(path2) if path2 else None)
    )

    final_df = (
        pd.merge(df2, df1, on="Patient File No.", suffixes=("", "_y"), how="left")
        if df2 is not None
        else df1
    )

    # Drop unwated & repeated features after merging
    final_df = final_df.loc[:, ~final_df.columns.str.contains(r"_y$|Unnamed")]

    print("✅ Data successfully loaded")

    return final_df


def fix_typo(df: pd.DataFrame, name_mapping: dict[str, str]) -> pd.DataFrame:
    df_result = df.copy()

    return df_result.rename(columns=name_mapping)


def adjust_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    df_result = df.copy()

    old_col_names = df_result.columns.tolist()
    new_col_names = [
        re.sub(
            r"([a-zA-Z])\(|\s+", lambda x: f"{x.group(1)} (" if x.group(1) else " ", col
        ).strip()
        for col in old_col_names
    ]

    df_result.columns = new_col_names

    return df_result


def drop_features(df: pd.DataFrame, target: list[str] | str) -> pd.DataFrame:
    df_result = df.copy()

    features = [target] if isinstance(target, str) else target

    return df_result.drop(features, axis=1, inplace=False, errors="ignore")


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


def fill_missing_val(
    df: pd.DataFrame, fmap: dict[str, Literal["mean", "median", "mode"] | int | float]
) -> pd.DataFrame:
    df_result = df.copy()

    for feature, fill_with in fmap.items():
        if feature not in df_result.columns:
            continue

        target_index = df_result[df_result[feature].isnull()].index

        if target_index.empty:
            continue

        if fill_with == "mean":
            val = df_result[feature].mean()
        elif fill_with == "median":
            val = df_result[feature].median()
        elif fill_with == "mode":
            mode_series = df_result.loc[df_result["PCOS (Y/N)"] == 0, feature].mode()
            val = (
                mode_series.iloc[0]
                if not mode_series.empty
                else df_result[feature].mode().iloc[0]
            )
        else:
            val = fill_with

        df_result.loc[target_index, feature] = val

    return df_result


def handle_extreme_val(
    df: pd.DataFrame,
    target: list[str] | str,
    threshold: float = 5.0,
    group_by: str | None = None,
) -> pd.DataFrame:
    df_result = df.copy()

    features = [target] if isinstance(target, str) else target

    for feature in features:
        if feature not in df.columns:
            continue

        med = df_result[feature].median()
        # Median Absolute Deviation (MAD)
        mad = (df_result[feature] - med).abs().median()

        # Avoid ZerDivisionError
        if mad == 0:
            continue

        # Source: https://medium.com/@fawwazmts/z-score-and-modified-z-score-f689296e4d3a
        mod_z_score = 0.6745 * (df_result[feature] - med).abs() / mad

        extreme_mask = mod_z_score > threshold

        if extreme_mask.any():
            clean_median = df_result.loc[~extreme_mask, feature].median()

            if group_by and group_by in df_result.columns:
                clean_df = df_result[~extreme_mask]
                group_medians = clean_df.groupby(group_by)[feature].median()

                replacement = (
                    df_result.loc[extreme_mask, group_by]
                    .map(group_medians)
                    .fillna(clean_median)
                )
                df_result.loc[extreme_mask, feature] = replacement
            else:
                df_result.loc[extreme_mask, feature] = clean_median

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
