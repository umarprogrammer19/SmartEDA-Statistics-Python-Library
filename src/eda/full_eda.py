from ..cleaning.missing_values import missing_percentage, fill_missing
from ..cleaning.outliers import detect_outliers_zscore
from ..features.relationships import numerical_relationship, categorical_relationship
from ..features.importance import top_features
from ..statistics.distributions import analyze_distribution


def full_eda(df, target):
    insights = {}

    # Missing values
    insights["missing"] = missing_percentage(df)

    # Handle missing
    df = fill_missing(df)

    # Target variable distribution analysis
    insights["target_distribution"] = analyze_distribution(df[target])

    # Outliers
    outlier_cols = {}
    for col in df.select_dtypes(include=["int64", "float64"]):
        outliers = detect_outliers_zscore(df[col])
        if len(outliers) > 0:
            outlier_cols[col] = len(outliers)
    insights["outliers"] = outlier_cols

    # Numerical relationships
    num_corr = numerical_relationship(df, target)
    insights["correlation"] = num_corr

    # Categorical relationships
    insights["categorical_relationships"] = categorical_relationship(df, target)

    # Top Features
    insights["top_features"] = top_features(num_corr)

    return df, insights
