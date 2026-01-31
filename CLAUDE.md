# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Smart EDA (Exploratory Data Analysis) Library written in Python that provides automated data analysis capabilities. The library performs comprehensive EDA including missing value analysis, outlier detection, correlation analysis, and feature importance assessment.

## Architecture & Structure

The project follows a modular structure organized into four main directories:

- `src/eda/`: Main EDA orchestration logic (`full_eda.py`)
- `src/cleaning/`: Data cleaning utilities (missing values, outliers, duplicates)
- `src/features/`: Feature analysis and relationship identification
- `src/statistics/`: Statistical functions and hypothesis testing utilities

The main entry point is `main.py` which loads a CSV dataset (`data.csv`) and performs automated EDA based on user-specified target column.

## Key Components

### Core Modules
- `main.py`: Entry point that orchestrates the EDA workflow
- `src/eda/full_eda.py`: Main EDA function that integrates all analysis modules
- `src/cleaning/missing_values.py`: Handles missing value detection and imputation
- `src/cleaning/outliers.py`: Detects outliers using Z-score and IQR methods
- `src/features/relationships.py`: Analyzes correlations between variables
- `src/features/importance.py`: Identifies top important features based on correlation

### Data Flow
1. Load dataset from `data.csv`
2. Perform missing value analysis and imputation
3. Detect and report outliers in numerical columns
4. Calculate correlations with target variable
5. Identify top important features
6. Return cleaned dataset and insights

## Development Commands

### Setup
```bash
uv sync  # Install dependencies using uv
```

### Running the EDA
```bash
python main.py  # Runs the automated EDA pipeline
```

### Dependencies
The project uses:
- pandas: For data manipulation
- numpy: For numerical computations
- scipy: For statistical functions

Dependencies are managed via `pyproject.toml` and `uv.lock`.

## Common Tasks

### Adding New Analysis Modules
- Place new modules in appropriate directories (`cleaning`, `features`, `statistics`)
- Follow the existing function signature patterns
- Import and integrate into `full_eda.py` as needed

### Extending EDA Functionality
- Modify `src/eda/full_eda.py` to include new analysis steps
- Add results to the `insights` dictionary to be reported to users
- Consider adding new submodules in the appropriate directories

### Testing
- Place test files alongside implementation files with `_test.py` suffix
- Use standard Python unittest or pytest frameworks