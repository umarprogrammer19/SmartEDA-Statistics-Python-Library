import numpy as np
from scipy.stats import norm


def pdf(x, mean=0, sd=1):
    return norm.pdf(x, mean, sd)


def cdf(x, mean=0, sd=1):
    return norm.cdf(x, mean, sd)


def analyze_distribution(data):
    """Analyze distribution of data: mean, std dev, skewness"""
    import pandas as pd
    import numpy as np
    from scipy import stats

    series = pd.Series(data)

    result = {
        'mean': series.mean(),
        'std_dev': series.std(),
        'skewness': series.skew(),
        'kurtosis': series.kurtosis(),
        'median': series.median(),
        'min': series.min(),
        'max': series.max()
    }

    return result
