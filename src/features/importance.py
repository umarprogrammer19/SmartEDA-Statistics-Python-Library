def top_features(correlation_series, n=10):
    return correlation_series.abs().sort_values(ascending=False).head(n)
