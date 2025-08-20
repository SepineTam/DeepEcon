# DeepEcon/transforms/drop

- [English](README.md)
- [简体中文](README.zh-CN.md)

Data filtering and selection utilities for managing dataset columns and rows.

## Classes

### DropVar
Remove specified columns from the dataset.

### KeepVar
Keep only specified columns, dropping all others.

### DropCondition
Remove rows based on a condition expression.

### KeepCondition
Keep only rows that satisfy a condition.

## Usage

```python
from deepecon.transforms import DropVar, DropCondition
from deepecon.core.condition import Condition

# Remove columns
df_clean = DropVar(df).transform(X_cols=['temp_column'])

# Filter rows
df_filtered = DropCondition(df).transform(
    _if_exp=Condition(lambda x: x['age'] >= 18)
)
```

## Parameters

| Class | Key Parameters | Description |
|-------|----------------|-------------|
| DropVar | `X_cols` | List of columns to remove |
| KeepVar | `X_cols` | List of columns to keep |
| DropCondition | `_if_exp` | Condition to filter rows |
| KeepCondition | `_if_exp` | Condition to keep rows |