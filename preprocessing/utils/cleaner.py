import re
from typing import Literal
import pandas as pd


def fix_typo(df: pd.DataFrame, name_mapping: dict[str, str]) -> pd.DataFrame:
    return df.copy().rename(columns=name_mapping)


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
    features = [target] if isinstance(target, str) else target
    return df.copy().drop(features, axis=1, errors="ignore")


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
        mad = (df_result[feature] - med).abs().median()

        # Avoid ZeroDivisionError
        if mad == 0:
            continue

        # Modified Z-Score calculation
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
