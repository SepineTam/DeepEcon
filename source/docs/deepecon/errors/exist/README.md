# DeepEcon/errors/exist

- [English](README.md)
- [简体中文](README.zh-CN.md)

Error types for when trying to create or access something that already exists.

## Error Types

- **FileExistError (3001)**: File already exists
- **VarExistError (3002)**: Variable/column already exists

## Quick Reference

| Error Code | Description | Example Fix |
|------------|-------------|-------------|
| 3001 | Check before creating file | `os.path.exists(filepath)` |
| 3002 | Use replace=True parameter | `transform(..., replace=True)` |

## Usage

```python
from deepecon.core.errors import VarExistError

try:
    # Attempt to create existing column
    pass
except VarExistError as e:
    print(f"Variable already exists: {e}")
    # Handle by either using replace=True or choosing different name
```