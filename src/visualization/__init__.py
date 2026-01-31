import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional
import os

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")


def plot_missing_values(df: pd.DataFrame, save_path: Optional[str] = None):
    """Plot missing value percentages for each column."""
    missing_pct = df.isnull().mean() * 100
    missing_pct = missing_pct[missing_pct > 0]  # Only show columns with missing values

    if len(missing_pct) == 0:
        print("No missing values to visualize.")
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing_pct.values, y=missing_pct.index, palette='viridis')
    plt.title('Missing Value Percentage by Column', fontsize=16, fontweight='bold')
    plt.xlabel('Percentage of Missing Values (%)', fontsize=12)
    plt.ylabel('Columns', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, 'missing_values.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_target_distribution(df: pd.DataFrame, target_col: str, save_path: Optional[str] = None):
    """Plot distribution of target variable."""
    plt.figure(figsize=(12, 5))

    # Histogram
    plt.subplot(1, 2, 1)
    if df[target_col].dtype in ['object', 'category']:
        # For categorical target, show value counts
        value_counts = df[target_col].value_counts()
        sns.barplot(x=value_counts.index.astype(str), y=value_counts.values)
        plt.title(f'Distribution of {target_col}', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
    else:
        # For numeric target, show histogram
        sns.histplot(data=df, x=target_col, kde=True)
        plt.title(f'Distribution of {target_col}', fontsize=14, fontweight='bold')

    # Box plot
    plt.subplot(1, 2, 2)
    if df[target_col].dtype in ['object', 'category']:
        # For categorical target, we can't make a box plot, so show value counts again
        value_counts = df[target_col].value_counts()
        sns.barplot(x=value_counts.index.astype(str), y=value_counts.values)
        plt.title(f'Value Counts of {target_col}', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
    else:
        sns.boxplot(y=df[target_col])
        plt.title(f'Box Plot of {target_col}', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, f'{target_col}_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, target_col: str, save_path: Optional[str] = None):
    """Plot correlation heatmap for numeric features."""
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['number'])

    if target_col in numeric_df.columns:
        # Only include columns that have correlation with the target
        correlations = numeric_df.corr()[target_col].abs().sort_values(ascending=False)
        relevant_cols = correlations.head(10).index.tolist()  # Top 10 correlated features

        # Create correlation matrix with only relevant columns
        corr_matrix = numeric_df[relevant_cols].corr()

        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                    square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title(f'Correlation Heatmap (Top Features)', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(os.path.join(save_path, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    else:
        print(f"Target column '{target_col}' is not numeric, skipping correlation heatmap.")


def plot_top_features(correlation_series: pd.Series, target_col: str, save_path: Optional[str] = None):
    """Plot top features based on correlation with target."""
    if correlation_series.empty or correlation_series.dtype == 'object':
        print("No correlation data to visualize.")
        return

    # Get top features excluding the target itself
    top_features = correlation_series.abs().sort_values(ascending=True)
    top_features = top_features[:-1]  # Exclude the target itself
    top_features = top_features.tail(10)  # Get top 10 features

    if len(top_features) == 0:
        print("No features to visualize.")
        return

    plt.figure(figsize=(10, 6))
    colors = plt.cm.RdYlBu_r(top_features.abs() / top_features.abs().max())
    bars = plt.barh(range(len(top_features)), top_features.values, color=colors)
    plt.yticks(range(len(top_features)), top_features.index)
    plt.xlabel(f'Absolute Correlation with {target_col}', fontsize=12)
    plt.title(f'Top Features Correlated with {target_col}', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, 'top_features.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_categorical_relationships(df: pd.DataFrame, target_col: str, save_path: Optional[str] = None):
    """Plot relationships between categorical features and target."""
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    cat_cols = [col for col in cat_cols if col != target_col]  # Exclude target column

    if len(cat_cols) == 0:
        print("No categorical columns to visualize.")
        return

    n_cols = min(2, len(cat_cols))
    n_rows = (len(cat_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_cols > 1 else [axes]
    else:
        axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        if df[target_col].dtype in ['object', 'category']:
            # If target is categorical, show count plot
            sns.countplot(data=df, x=col, ax=axes[i])
            axes[i].set_title(f'Count of {col}')
            axes[i].tick_params(axis='x', rotation=45)
        else:
            # If target is numeric, show box plot
            sns.boxplot(data=df, x=col, y=target_col, ax=axes[i])
            axes[i].set_title(f'{target_col} by {col}')
            axes[i].tick_params(axis='x', rotation=45)

    # Hide unused subplots
    for j in range(len(cat_cols), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, 'categorical_relationships.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_outliers(df: pd.DataFrame, target_col: str, save_path: Optional[str] = None):
    """Plot outliers in numerical features."""
    numeric_cols = df.select_dtypes(include=['number']).columns
    numeric_cols = [col for col in numeric_cols if col != target_col]  # Exclude target

    if len(numeric_cols) == 0:
        print("No numeric columns to visualize for outliers.")
        return

    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if len(numeric_cols) == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_cols > 1 else [axes]
    else:
        axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.boxplot(y=df[col], ax=axes[i])
        axes[i].set_title(f'Outliers in {col}')
        axes[i].tick_params(axis='x', rotation=45)

    # Hide unused subplots
    for j in range(len(numeric_cols), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, 'outliers.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def generate_visualizations(df: pd.DataFrame, target_col: str, save_path: Optional[str] = "visualizations"):
    """Generate all visualizations and save them to a directory."""
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    print("Generating visualizations...")

    # 1. Missing values plot
    plot_missing_values(df, save_path)

    # 2. Target distribution plot
    plot_target_distribution(df, target_col, save_path)

    # 3. Correlation heatmap
    plot_correlation_heatmap(df, target_col, save_path)

    # 4. Top features plot
    if df[target_col].dtype in ['int64', 'float64']:
        numeric_df = df.select_dtypes(include=['number'])
        if target_col in numeric_df.columns:
            correlations = numeric_df.corr()[target_col]
            plot_top_features(correlations, target_col, save_path)

    # 5. Categorical relationships plot
    plot_categorical_relationships(df, target_col, save_path)

    # 6. Outliers plot
    plot_outliers(df, target_col, save_path)

    if save_path:
        print(f"All visualizations saved to '{save_path}' directory.")
    else:
        print("Visualization generation complete.")