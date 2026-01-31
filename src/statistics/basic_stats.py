import numpy as np
import pandas as pd
from collections import Counter


def mean(data):
    data = np.array(data)
    return np.mean(data)


def median(data):
    data = np.array(data)
    return np.median(data)


def mode(data):
    data = list(data)
    return Counter(data).most_common(1)[0][0]


def variance(data):
    return np.var(data, ddof=1)


def std_dev(data):
    return np.std(data, ddof=1)


def z_score(value, data):
    mean_val = mean(data)
    std_val = std_dev(data)
    return (value - mean_val) / std_val
