# DeepEcon Error Handling System

Comprehensive error handling and documentation system for DeepEcon package.

- [English](README.md)
- [简体中文](README.zh-CN.md)

## Overview

The errors module provides a structured error handling system with detailed error codes, documentation links, and logging capabilities. Each error type includes specific error codes and links to detailed documentation.

## Error Categories

### 1. Not Found Errors (1000-1999)
- **[FileNotFoundError](not_found/README.md)**: Files not found (1001)
- **[VarNotFoundError](not_found/README.md)**: Variables/columns not found (1002)
- **[ConditionNotFoundError](not_found/README.md)**: Missing condition expressions (1003)
- **[OperatorNotFoundError](not_found/README.md)**: Invalid or missing operators (1004)

### 2. Length Mismatch Errors (2000-2999)
- **[LengthNotMatchError](length/README.md)**: Array dimension mismatches (2001)

### 3. Existence Errors (3000-3999)
- **[FileExistError](exist/README.md)**: Files already exist (3001)
- **[VarExistError](exist/README.md)**: Variables/columns already exist (3002)

## Error Code Reference

| Error Code | Error Type | Description |
|------------|------------|-------------|
| 1001 | FileNotFoundError | File does not exist |
| 1002 | VarNotFoundError | Variable/column not found |
| 1003 | ConditionNotFoundError | Missing condition expression |
| 1004 | OperatorNotFoundError | Invalid or missing operator |
| 2001 | LengthNotMatchError | Array dimension mismatch |
| 3001 | FileExistError | File already exists |
| 3002 | VarExistError | Variable/column already exists |

## Quick Examples

```python
from deepecon.core.errors import (
    VarNotFoundError,
    LengthNotMatchError,
    FileNotFoundError
)

try:
    # Your data processing code
    result = some_transform(df)
except VarNotFoundError as e:
    print(f"Variable error: {e}")
except LengthNotMatchError as e:
    print(f"Dimension error: {e}")
except FileNotFoundError as e:
    print(f"File error: {e}")
```

## Documentation Links

Each error provides direct links to detailed documentation:
- **Not Found Errors**: [Documentation](not_found/README.md)
- **Length Errors**: [Documentation](length/README.md)
- **Existence Errors**: [Documentation](exist/README.md)

## Usage

All errors inherit from `ErrorBase` providing:
- Automatic logging
- Documentation links
- Error codes
- Web browser integration
