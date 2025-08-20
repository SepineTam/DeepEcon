# DeepEcon/transforms/corr

- [English](README.md)
- [简体中文](README.zh-CN.md)

Correlation analysis utilities for measuring relationships between variables.

## Classes

### PearsonCorr
Calculate Pearson correlation coefficients between variables.

**Methods:**
- **Array method**: Correlation matrix for multiple variables
- **y-x method**: Correlation between dependent and independent variables

## Usage

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
    X_cols=['age', 'education', 'experience']
)
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `y_col` | str | Target variable (for y-x method) |
| `X_cols` | List[str] | Independent variables |
| `is_array` | bool | Whether to compute correlation matrix |