from scipy.stats import ttest_ind, chisquare, norm


def z_test(sample_mean, pop_mean, std, n):
    z = (sample_mean - pop_mean) / (std / (n**0.5))
    p = 2 * (1 - norm.cdf(abs(z)))
    return z, p


def t_test(sample1, sample2):
    return ttest_ind(sample1, sample2)


def chi_square(observed, expected):
    return chisquare(observed, f_exp=expected)
