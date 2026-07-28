import pandas as pd

from preprocessing.helper import (
    adjust_whitespace,
    cast_type,
    convert_to_numeric,
    create_new_features,
    drop_features,
    fill_missing_val,
    fix_typo,
    handle_extreme_val,
    repos_col,
    standardize_col_names,
)


def run_preprocessing(data) -> pd.DataFrame:
    df = drop_features(df=data, target=["Patient File No.", "Sl. No"])

    df = fix_typo(
        df=df,
        name_mapping={
            "Marraige Status (Yrs)": "Marriage Status (Yrs)",
            "No. of aborptions": "No. of abortions",
        },
    )
    df = adjust_whitespace(df=df)

    df = convert_to_numeric(df=df, errors="coerce")

    df = fill_missing_val(
        df=df,
        fmap={
            "II beta-HCG (mIU/mL)": 1.99,
            "AMH (ng/mL)": "median",
            "Fast food (Y/N)": "mode",
            "Marriage Status (Yrs)": "median",
        },
    )

    df = handle_extreme_val(
        df=df,
        target=["LH (mIU/mL)", "FSH (mIU/mL)"],
        threshold=3.5,
        group_by="Pregnant (Y/N)",
    )
    df = create_new_features(
        df=df,
        fmap=[
            {
                "name": "LH/FSH Ratio",
                "features": ["LH (mIU/mL)", "FSH (mIU/mL)"],
                "how": "div",
            },
            {
                "name": "Total Follicles",
                "features": ["Follicle No. (L)", "Follicle No. (R)"],
                "how": "add",
            },
            {
                "name": "Follicles Difference",
                "features": ["Follicle No. (L)", "Follicle No. (R)"],
                "how": "sub",
            },
        ],
    )

    df = drop_features(
        df=df,
        target=["FSH/LH", "Blood Group", "Marriage Status (Yrs)", "No. of abortions"],
    )

    df = cast_type(df=df, cols="Fast food (Y/N)", target_type="int64")

    df = repos_col(df=df)

    df = standardize_col_names(df=df)

    return df
