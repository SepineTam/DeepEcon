# DeepEcon Transforms
- [English](README.md)
- [简体中文](README.zh-CN.md)

Data transformation and preprocessing utilities for econometric analysis.

## Overview

The transforms module provides a comprehensive set of data transformation tools designed for econometric analysis. These transforms include mathematical operations, data cleaning, correlation analysis, and statistical transformations.

## Module Structure

```
deepcon/transforms/
├── __init__.py          # Public API exports
├── basic/               # Basic mathematical operations
│   ├── __init__.py
│   ├── base.py         # Base classes for mathematical transforms
│   └── summarize.py    # Statistical summarization
├── corr/               # Correlation analysis
│   ├── __init__.py
│   ├── base.py         # Base correlation classes
│   ├── pearson.py      # Pearson correlation
│   ├── spearman.py     # Spearman rank correlation
│   ├── kendall.py      # Kendall's tau correlation
│   ├── cramer_v.py     # Cramér's V correlation
│   ├── phi.py          # Phi coefficient
│   ├── distance.py     # Distance correlation
│   └── point_biserial.py # Point-biserial correlation
├── drop.py             # Data filtering and selection
└── winsor2.py          # Winsorization for outlier treatment
```

## Available Transforms

### 1. Basic Operations (`basic`)

#### BasicMath
Base class for mathematical transformations on data columns.

**Features:**
- Support for custom mathematical expressions
- Column creation and replacement
- Conditional execution support

#### Summarize
Generate comprehensive statistical summaries for data columns.

**Statistical Measures:**
- Variable name (Var)
- Sample size (N)
- Mean, Standard deviation
- Minimum, Maximum values
- Quartiles (Q1, Q3)
- Median
- Missing value count
- Unique value count

**Usage Example:**
```python
from deepecon.transforms import Summarize

summarizer = Summarize(df)
summary = summarizer.transform(
    X_cols=['income', 'age', 'education'],
    summ_cols=['Var', 'N', 'Mean', 'Std', 'Min', 'Max']
)
```

### 2. Correlation Analysis (`corr`)

#### PearsonCorr
Calculate Pearson correlation coefficients between variables.

**Methods:**
- **Array method**: Correlation matrix for multiple variables
- **y-x method**: Correlation between one dependent and multiple independent variables

**Usage Example:**
```python
from deepecon.transforms import PearsonCorr

# Correlation matrix
corr_matrix = PearsonCorr(df).transform(
    X_cols=['var1', 'var2', 'var3'],
    is_array=True
)

# y vs x correlations
correlations = PearsonCorr(df).transform(
    y_col='income',
    X_cols=['education', 'experience', 'age']
)
```

### 3. Data Filtering (`drop`)

#### DropVar
Remove specified columns from the dataset.

**Parameters:**
- `X_cols`: List of column names to drop
- `_if_exp`: Conditional expression for filtering

#### KeepVar
Keep only specified columns, dropping all others.

**Parameters:**
- `X_cols`: List of column names to keep
- `_if_exp`: Conditional expression for filtering

#### DropCondition
Remove rows based on a condition.

**Parameters:**
- `_if_exp`: Condition expression for row filtering

#### KeepCondition
Keep only rows that satisfy a condition.

**Parameters:**
- `_if_exp`: Condition expression for row filtering

**Usage Example:**
```python
from deepecon.transforms import DropCondition, KeepVar
from deepecon.core.condition import Condition

# Drop rows where age < 18
df_filtered = DropCondition(df).transform(
    _if_exp=Condition(lambda x: x['age'] < 18)
)

# Keep only income and education columns
df_subset = KeepVar(df).transform(X_cols=['income', 'education'])
```

### 4. Outlier Treatment (`winsor2`)

#### Winsor2
Winsorize data to handle outliers by setting extreme values to specified percentiles.

**Parameters:**
- `p`: Tuple of lower and upper percentiles (default: (0.01, 0.99))
- `suffix`: Suffix for new column names (default: '_w')
- `replace`: Whether to replace original columns

**Usage Example:**
```python
from deepecon.transforms import Winsor2

# Winsorize income and age variables
df_winsorized = Winsor2(df).transform(
    X_cols=['income', 'age'],
    p=(0.05, 0.95),
    suffix='_winsorized',
    replace=False
)
```

## API Reference

### Public Exports

Available directly from `deepecon.transforms`:

- `BasicMath`: Base mathematical operations
- `Summarize`: Statistical summarization
- `PearsonCorr`: Pearson correlation analysis
- `DropVar`: Column removal
- `KeepVar`: Column selection
- `DropCondition`: Row filtering by condition
- `KeepCondition`: Row selection by condition

### Advanced Usage

All transforms inherit from `TransformBase` and support:

- **Conditional execution** via `_if_exp` parameter
- **Column selection** via `X_cols` parameter
- **In-place modification** via `replace` parameter
- **Method chaining** for complex data pipelines

## Best Practices

1. **Data Validation**: Always validate input data before transformation
2. **Column Selection**: Use explicit column lists for better performance
3. **Conditional Logic**: Leverage conditions for targeted transformations
4. **Memory Management**: Use `replace=True` for in-place operations to save memory
5. **Pipeline Design**: Chain transformations for complex preprocessing workflows

## Error Handling

The transforms module provides comprehensive error handling:

- `VarNotFoundError`: When specified columns don't exist
- `ConditionNotFoundError`: When required conditions are missing
- `LengthNotMatchError`: When data dimensions are incompatible
- `OperatorNotFoundError`: When mathematical operators are invalid

## Examples

### Complete Data Processing Pipeline

```python
from deepecon.transforms import *
from deepecon.core.condition import Condition

# Create processing pipeline
df_processed = (
    Winsor2(df)
    .transform(X_cols=['income', 'age'], p=(0.01, 0.99))
)

df_processed = (
    DropCondition(df_processed)
    .transform(_if_exp=Condition(lambda x: x['age'] >= 18))
)

# Generate summary statistics
summary = Summarize(df_processed).transform(X_cols=['income', 'age'])

# Calculate correlations
correlations = PearsonCorr(df_processed).transform(
    y_col='income',
    X_cols=['age', 'education', 'experience']
)
```