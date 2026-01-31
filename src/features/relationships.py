import pandas as pd


def numerical_relationship(df, target):
    return df.corr()[target].sort_values(ascending=False)


def categorical_relationship(df, target):
    result = {}
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in cat_cols:
        stat = df.groupby(col)[target].mean()
        result[col] = stat
    return result
