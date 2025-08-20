# DeepEcon/errors

- [English](README.md)
- [简体中文](README.zh-CN.md)

Comprehensive error handling system with structured error codes and documentation links.

## Error Categories

### Not Found Errors (1000-1999)
- **[not_found](not_found/README.md)**: Files, variables, conditions, operators not found

### Length Errors (2000-2999)  
- **[length](length/README.md)**: Array and dimension mismatches

### Existence Errors (3000-3999)
- **[exist](exist/README.md)**: Files and variables that already exist

## Error Code Reference

| Code | Type | Description |
|------|------|-------------|
| 1001 | FileNotFoundError | File does not exist |
| 1002 | VarNotFoundError | Variable/column not found |
| 1003 | ConditionNotFoundError | Missing condition expression |
| 1004 | OperatorNotFoundError | Invalid or missing operator |
| 2001 | LengthNotMatchError | Array dimension mismatch |
| 3001 | FileExistError | File already exists |
| 3002 | VarExistError | Variable/column already exists |

## Usage

```python
from deepecon.core.errors import VarNotFoundError

try:
    # Your data processing code
    pass
except VarNotFoundError as e:
    print(f"Variable not found: {e}")
```
