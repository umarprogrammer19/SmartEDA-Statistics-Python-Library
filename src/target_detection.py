"""
Helper functions for target variable detection
"""

def detect_target_variable(df):
    """
    Automatically detect the most suitable target variable in a dataset.

    The algorithm prioritizes potential target variables based on:
    1. Columns with high cardinality (more unique values) - likely to be targets
    2. Numeric columns (usually regression targets)
    3. Categorical columns with reasonable cardinality (classification targets)
    4. Columns with names that suggest they might be targets ('target', 'outcome', 'result', etc.)

    Args:
        df: Input dataframe

    Returns:
        str: Name of the detected target column, or None if none found
    """
    # Define potential target column names (case-insensitive)
    potential_target_names = [
        'target', 'outcome', 'result', 'prediction', 'predict', 'class', 'label',
        'y', 'response', 'dependent', 'output', 'value', 'performance', 'index'
    ]

    # First, look for columns with target-like names
    for col in df.columns:
        col_lower = col.lower()
        if any(target_name in col_lower for target_name in potential_target_names):
            return col

    # If no target-like names found, prioritize columns by characteristics
    scores = {}
    for col in df.columns:
        score = 0

        # Score based on data type (numeric columns get priority)
        if df[col].dtype in ['int64', 'float64']:
            score += 3
        elif df[col].dtype == 'object':
            # Check cardinality for categorical columns
            unique_ratio = df[col].nunique() / len(df)
            if 0.1 <= unique_ratio <= 0.8:  # Reasonable cardinality for classification
                score += 2
        else:
            score += 1

        # Score based on cardinality (high cardinality might indicate target)
        unique_count = df[col].nunique()
        if unique_count == len(df):  # All unique - might be ID
            score -= 2
        elif unique_count == 1:  # Only one value - not useful as target
            score -= 3
        elif unique_count >= len(df) * 0.8:  # High cardinality
            score += 2
        elif unique_count <= 2:  # Binary classification
            score += 1
        elif 2 < unique_count <= 10:  # Multi-class classification
            score += 1

        scores[col] = score

    # Return the column with highest score
    if scores:
        best_col = max(scores, key=scores.get)
        return best_col

    return None


def suggest_target_variables(df, top_n=3):
    """
    Suggest top N potential target variables from the dataset.

    Args:
        df: Input dataframe
        top_n: Number of suggestions to return (default 3)

    Returns:
        list: List of tuples (column_name, score) ranked by score descending
    """
    scores = {}
    for col in df.columns:
        score = 0

        # Score based on data type (numeric columns get priority)
        if df[col].dtype in ['int64', 'float64']:
            score += 3
        elif df[col].dtype == 'object':
            # Check cardinality for categorical columns
            unique_ratio = df[col].nunique() / len(df)
            if 0.1 <= unique_ratio <= 0.8:  # Reasonable cardinality for classification
                score += 2
        else:
            score += 1

        # Score based on cardinality
        unique_count = df[col].nunique()
        if unique_count == len(df):  # All unique - might be ID
            score -= 2
        elif unique_count == 1:  # Only one value - not useful as target
            score -= 3
        elif unique_count >= len(df) * 0.8:  # High cardinality
            score += 2
        elif unique_count <= 2:  # Binary classification
            score += 1
        elif 2 < unique_count <= 10:  # Multi-class classification
            score += 1

        scores[col] = score

    # Sort by score in descending order and return top N
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores[:top_n]