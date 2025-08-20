# DeepEcon/transforms/winsor2

- [English](README.md)
- [简体中文](README.zh-CN.md)

Outlier treatment through winsorization by setting extreme values to specified percentiles.

## Classes

### Winsor2
Winsorize data to handle outliers by capping extreme values at specified percentiles.

**Features:**
- Configurable percentile bounds
- Optional column replacement
- Custom suffix for new columns
- Conditional execution support

## Usage

```python
from deepecon.transforms import Winsor2

# Basic winsorization
df_winsorized = Winsor2(df).transform(
    X_cols=['income', 'age'],
    p=(0.01, 0.99)
)

# With custom suffix
df_winsorized = Winsor2(df).transform(
    X_cols=['income'],
    p=(0.05, 0.95),
    suffix='_winsor',
    replace=False
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X_cols` | List[str] | - | Columns to winsorize |
| `p` | Tuple[float, float] | (0.01, 0.99) | Lower and upper percentiles |
| `suffix` | str | "_w" | Suffix for new column names |
| `replace` | bool | False | Whether to replace original columns |