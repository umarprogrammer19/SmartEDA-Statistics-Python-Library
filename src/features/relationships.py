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

    # Remove the target column from categorical columns to avoid self-analysis
    cat_cols = [col for col in cat_cols if col != target]

    for col in cat_cols:
        # Check if target is numeric to decide what aggregation to use
        if df[target].dtype in ['int64', 'float64']:
            # For numeric target, calculate mean
            stat = df.groupby(col)[target].mean()
        else:
            # For categorical target, calculate value counts for each group
            stat = df.groupby(col)[target].value_counts()
        result[col] = stat
    return result
