import pandas as pd


def missing_percentage(df):
    return df.isnull().mean() * 100


def fill_missing(df, method="mean"):
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            # Check if column is numeric (not string/object)
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                if method == "mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif method == "median":
                    df[col] = df[col].fillna(df[col].median())
                elif method == "mode":
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else df[col].iloc[0] if len(df[col]) > 0 else '')
            else:  # For non-numeric columns, use mode
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else df[col].iloc[0] if len(df[col]) > 0 else '')
    return df


def remove_missing_rows(df):
    return df.dropna()
