# DeepEcon/transforms/basic

- [English](README.md)
- [简体中文](README.zh-CN.md)

Basic mathematical operations and statistical summarization for data transformation.

## Classes

### BasicMath
Base class for mathematical transformations on data columns.

**Features:**
- Custom mathematical expressions
- Column creation and replacement
- Conditional execution support

### Summarize  
Generate comprehensive statistical summaries for data columns.

**Statistics Available:**
- Variable name (Var)
- Sample size (N)
- Mean, standard deviation
- Min, max values
- Quartiles (Q1, Q3)
- Median
- Missing value count
- Unique value count

## Usage

```python
from deepecon.transforms import Summarize

# Generate summary statistics
summary = Summarize(df).transform(
    X_cols=['income', 'age'],
    summ_cols=['Var', 'N', 'Mean', 'Std']
)
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `X_cols` | List[str] | Columns to process |
| `summ_cols` | List[str] | Statistics to compute |