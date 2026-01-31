import pandas as pd


def missing_percentage(df):
    return df.isnull().mean() * 100


def fill_missing(df, method="mean"):
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if method == "mean" and df[col].dtype != object:
                df[col] = df[col].fillna(df[col].mean())
            elif method == "median" and df[col].dtype != object:
                df[col] = df[col].fillna(df[col].median())
            elif method == "mode":
                df[col] = df[col].fillna(df[col].mode()[0])
    return df


def remove_missing_rows(df):
    return df.dropna()
