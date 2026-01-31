import numpy as np
from scipy.stats import norm


def pdf(x, mean=0, sd=1):
    return norm.pdf(x, mean, sd)


def cdf(x, mean=0, sd=1):
    return norm.cdf(x, mean, sd)
