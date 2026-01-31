import numpy as np


def detect_outliers_zscore(series):
    z = np.abs((series - series.mean()) / series.std())
    return series[z > 3]


def detect_outliers_iqr(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return series[(series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)]


def remove_outliers_zscore(df, columns, threshold=3):
    """Remove outliers using Z-score method"""
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns:
            z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
            df_clean = df_clean[z_scores <= threshold]
    return df_clean


def cap_outliers_iqr(df, columns):
    """Cap outliers using IQR method (winsorizing)"""
    df_capped = df.copy()
    for col in columns:
        if col in df_capped.columns:
            q1 = df_capped[col].quantile(0.25)
            q3 = df_capped[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df_capped[col] = df_capped[col].clip(lower=lower_bound, upper=upper_bound)
    return df_capped
