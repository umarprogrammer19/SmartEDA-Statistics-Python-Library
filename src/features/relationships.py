import pandas as pd


def numerical_relationship(df, target):
    # Select only numeric columns for correlation calculation
    numeric_df = df.select_dtypes(include=['number'])
    # Only compute correlations if the target column is numeric
    if target in numeric_df.columns:
        return numeric_df.corr()[target].sort_values(ascending=False)
    else:
        # If target is not numeric, we can't correlate with it, return empty series
        return pd.Series(dtype=float)


def categorical_relationship(df, target):
    result = {}
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in cat_cols:
        stat = df.groupby(col)[target].mean()
        result[col] = stat
    return result
