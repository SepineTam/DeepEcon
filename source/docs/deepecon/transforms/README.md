# DeepEcon/transforms

- [English](README.md)
- [简体中文](README.zh-CN.md)

Data transformation and preprocessing modules for econometric analysis.

## Module Structure

- **[basic](basic/README.md)**: Basic mathematical operations and summaries
- **[corr](corr/README.md)**: Correlation analysis utilities  
- **[drop](drop/README.md)**: Data filtering and selection
- **[winsor2](winsor2/README.md)**: Outlier treatment via winsorization

## Available Transforms

| Module | Classes | Purpose |
|--------|---------|---------|
| `basic` | BasicMath, Summarize | Mathematical operations & statistics |
| `corr` | PearsonCorr | Correlation analysis |
| `drop` | DropVar, KeepVar, DropCondition, KeepCondition | Data filtering |
| `winsor2` | Winsor2 | Outlier treatment |

## Quick Usage

```python
from deepecon.transforms import Summarize, PearsonCorr

# Basic summarization
summary = Summarize(df).transform(X_cols=['income', 'age'])

# Correlation analysis
corrs = PearsonCorr(df).transform(y_col='income', X_cols=['age', 'education'])
```

## API Reference

All transforms inherit from `TransformBase` providing:
- Conditional execution with `_if_exp`
- Column selection with `X_cols`
- In-place modification with `replace`
- Error handling with specific error codes