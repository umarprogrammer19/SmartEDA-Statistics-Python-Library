import numpy as np


def detect_outliers_zscore(series):
    z = np.abs((series - series.mean()) / series.std())
    return series[z > 3]


def detect_outliers_iqr(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return series[(series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)]
