# Smart EDA Library

A comprehensive Python library for automated Exploratory Data Analysis (EDA) and statistical analysis. Designed to help data scientists and AI practitioners quickly understand data quality, feature relationships, and prepare datasets for machine learning models.

## 🚀 Features

- **Automated EDA**: Perform comprehensive analysis with a single function call
- **Statistical Analysis**: Mean, median, mode, variance, standard deviation, and advanced statistics
- **Distribution Analysis**: Normal distribution functions, skewness, kurtosis
- **Hypothesis Testing**: Z-test, T-test, Chi-square test
- **Data Cleaning**: Missing value detection/handling, outlier detection/removal
- **Feature Analysis**: Correlation analysis, feature importance ranking
- **Rich CLI Interface**: Colorful, interactive command-line interface
- **Mixed Data Types**: Handles both numeric and categorical variables
- **Visualizations**: Comprehensive plots and charts using matplotlib and seaborn

## 📋 Requirements

- Python 3.13+
- Dependencies managed with `uv`

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd sn-smarteda
```

2. Install dependencies using uv:
```bash
uv sync
```

## 📊 Usage

### Command Line Interface

Place your dataset as `data.csv` in the project root and run:

```bash
# Interactive mode
python main.py

# Or specify target column directly
python main.py "target_column_name"

# With visualizations (using --viz or -v flag)
python main.py "target_column_name" --viz
# or
python main.py "target_column_name" -v
```

### Programmatic Usage

```python
import pandas as pd
from src.eda.full_eda import full_eda

# Load your dataset
df = pd.read_csv("your_dataset.csv")

# Run full EDA analysis
clean_df, insights = full_eda(df, target="your_target_column")

# Access insights
print(insights["missing"])          # Missing value percentages
print(insights["correlation"])      # Correlations with target
print(insights["top_features"])     # Top important features
print(insights["outliers"])         # Outlier detection results

# Generate visualizations
clean_df, insights = full_eda(df, target="your_target_column", generate_viz=True, viz_save_path="my_plots")
```

### Visualization Capabilities

The library generates the following visualizations:

- **Missing Values Plot**: Bar chart showing percentage of missing values per column
- **Target Distribution Plot**: Histogram and box plot for numeric targets, bar chart for categorical targets
- **Correlation Heatmap**: Heatmap showing correlations between top features
- **Top Features Plot**: Horizontal bar chart showing feature importance based on correlation
- **Categorical Relationships Plot**: Box plots showing relationship between categorical features and target
- **Outliers Plot**: Box plots for detecting outliers in numerical features

## 🏗️ Project Structure

```
src/
├── cleaning/           # Data cleaning utilities
│   ├── missing_values.py     # Missing value handling
│   ├── outliers.py          # Outlier detection and handling
│   └── duplicates.py        # Duplicate detection
├── eda/                # Main EDA orchestration
│   └── full_eda.py         # Complete EDA pipeline
├── features/           # Feature analysis
│   ├── relationships.py     # Feature relationships
│   └── importance.py        # Feature importance ranking
├── statistics/         # Statistical functions
│   ├── basic_stats.py       # Mean, median, mode, etc.
│   ├── distributions.py     # Distribution functions
│   └── hypothesis_tests.py  # Statistical tests
└── visualization/      # Visualization capabilities
    └── __init__.py         # Plotting functions and visualization generator
```

## 🔧 Available Functions

### Statistics Module
- `mean(data)`, `median(data)`, `mode(data)`
- `variance(data)`, `std_dev(data)`
- `z_score(value, data)`
- `pdf(x, mean, sd)`, `cdf(x, mean, sd)`
- `analyze_distribution(data)` - Comprehensive distribution analysis

### Data Cleaning Module
- `missing_percentage(df)` - Calculate missing value percentages
- `fill_missing(df, method="mean")` - Fill missing values
- `detect_outliers_zscore(series)`, `detect_outliers_iqr(series)`
- `remove_outliers_zscore(df, columns)`, `cap_outliers_iqr(df, columns)`
- `detect_duplicates(df)`, `remove_duplicates(df)`

### Feature Analysis Module
- `numerical_relationship(df, target)` - Correlation analysis
- `categorical_relationship(df, target)` - Categorical relationships
- `top_features(correlation_series, n=10)` - Feature importance ranking

## 📈 EDA Insights

The `full_eda()` function returns comprehensive insights including:

- **Missing Values**: Percentage of missing values per column
- **Target Distribution**: Mean, std dev, skewness, kurtosis for numeric targets
- **Outliers**: Detection results for numerical columns
- **Correlations**: Relationships between features and target
- **Top Features**: Ranked list of most important features
- **Categorical Relationships**: Group statistics for categorical features

## 💡 Example Output

```
+-------------------+
| SMART EDA LIBRARY |
+-------------------+
AUTO EDA STARTED

[OK] Dataset loaded successfully.

Columns in your dataset:
                   Dataset Columns
+----------------------------------------------------+
| #   | Column Name                      | Data Type |
|-----+----------------------------------+-----------|
| 1   | Hours Studied                    |   float64 |
| 2   | Previous Scores                  |   float64 |
| 3   | Extracurricular Activities       |       str |
| 4   | Sleep Hours                      |   float64 |
| 5   | Sample Question Papers Practiced |   float64 |
| 6   | Performance Index                |   float64 |
+----------------------------------------------------+

Using target column from command line: Performance Index

Running full EDA on target: Performance Index

EDA Insights Summary:

+---------------------------------- MISSING ----------------------------------+
| Hours Studied                       0.24                                    |
| Previous Scores                     0.24                                    |
| Extracurricular Activities          0.24                                    |
| Sleep Hours                         0.24                                    |
| Sample Question Papers Practiced    0.24                                    |
| Performance Index                   0.24                                    |
+-----------------------------------------------------------------------------+
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support, please open an issue in the repository or contact the maintainers.