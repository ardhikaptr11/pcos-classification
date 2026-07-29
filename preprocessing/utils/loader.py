import pandas as pd


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
