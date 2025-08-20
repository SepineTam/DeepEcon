# DeepEcon/errors/length

- [English](README.md)
- [简体中文](README.zh-CN.md)

Error types for when data dimensions don't match expected requirements.

## Error Types

- **LengthNotMatchError (2001)**: Array or dimension mismatch

## Quick Reference

| Error Code | Description | Example Fix |
|------------|-------------|-------------|
| 2001 | Ensure compatible dimensions | `len(X_cols) >= 2` for correlation |

## Usage

```python
from deepecon.core.errors import LengthNotMatchError

try:
    # Attempt operation with mismatched dimensions
    pass
except LengthNotMatchError as e:
    print(f"Dimension mismatch: {e}")
    # Ensure arrays have compatible lengths
```

## Common Cases

- **Correlation matrices**: Require at least 2 variables
- **Array operations**: Require matching dimensions
- **Statistical functions**: Require minimum data points