from cleaning.missing_values import missing_percentage, fill_missing
from cleaning.outliers import detect_outliers_zscore
from features.relationships import numerical_relationship
from features.importance import top_features


def full_eda(df, target):
    insights = {}

    # Missing values
    insights["missing"] = missing_percentage(df)

    # Handle missing
    df = fill_missing(df)

    # Outliers
    outlier_cols = {}
    for col in df.select_dtypes(include=["int", "float"]):
        outliers = detect_outliers_zscore(df[col])
        if len(outliers) > 0:
            outlier_cols[col] = len(outliers)
    insights["outliers"] = outlier_cols

    # Correlation
    corr = numerical_relationship(df, target)
    insights["correlation"] = corr

    # Top Features
    insights["top_features"] = top_features(corr)

    return df, insights
