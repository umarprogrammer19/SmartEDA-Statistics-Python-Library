import pandas as pd
from ..cleaning.missing_values import missing_percentage, fill_missing
from ..cleaning.outliers import detect_outliers_zscore
from ..features.relationships import numerical_relationship, categorical_relationship
from ..features.importance import top_features
from ..statistics.distributions import analyze_distribution
from ..visualization import generate_visualizations


def full_eda(df, target, generate_viz=False, viz_save_path="visualizations"):
    """
    Perform full EDA on a dataset.

    Args:
        df: Input dataframe
        target: Target column name
        generate_viz: Whether to generate visualizations (default False)
        viz_save_path: Path to save visualizations (default "visualizations")

    Returns:
        tuple: (cleaned_dataframe, insights_dict)
    """
    insights = {}

    # Missing values
    insights["missing"] = missing_percentage(df)

    # Handle missing
    df = fill_missing(df)

    # Target variable distribution analysis (only for numeric columns)
    if df[target].dtype in ['int64', 'float64']:
        insights["target_distribution"] = analyze_distribution(df[target])
    else:
        # For categorical targets, provide basic info about the target
        insights["target_distribution"] = {
            'dtype': str(df[target].dtype),
            'unique_count': df[target].nunique(),
            'unique_values': df[target].unique().tolist(),
            'value_counts': df[target].value_counts().to_dict()
        }

    # Outliers (only for numeric columns)
    outlier_cols = {}
    for col in df.select_dtypes(include=["int64", "float64"]):
        outliers = detect_outliers_zscore(df[col])
        if len(outliers) > 0:
            outlier_cols[col] = len(outliers)
    insights["outliers"] = outlier_cols

    # Numerical relationships (only if target is numeric)
    if df[target].dtype in ['int64', 'float64']:
        num_corr = numerical_relationship(df, target)
        insights["correlation"] = num_corr
        insights["top_features"] = top_features(num_corr)
    else:
        # For categorical targets, we can't compute correlation, so return empty
        insights["correlation"] = pd.Series(dtype=float)
        insights["top_features"] = pd.Series(dtype=float)

    # Categorical relationships
    insights["categorical_relationships"] = categorical_relationship(df, target)

    # Generate visualizations if requested
    if generate_viz:
        generate_visualizations(df, target, viz_save_path)

    return df, insights
