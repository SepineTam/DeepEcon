# DeepEcon 错误处理系统

DeepEcon包的全面错误处理和文档系统。

- [English](README.md)
- [简体中文](README.zh-CN.md)

## 概述

错误模块提供了结构化的错误处理系统，包含详细的错误代码、文档链接和日志记录功能。每种错误类型都包含特定的错误代码和详细文档链接。

## 错误类别

### 1. 未找到错误 (1000-1999)
- **[FileNotFoundError](not_found/README.md)**：文件未找到 (1001)
- **[VarNotFoundError](not_found/README.md)**：变量/列未找到 (1002)
- **[ConditionNotFoundError](not_found/README.md)**：缺少条件表达式 (1003)
- **[OperatorNotFoundError](not_found/README.md)**：无效或缺少运算符 (1004)

### 2. 长度不匹配错误 (2000-2999)
- **[LengthNotMatchError](length/README.md)**：数组维度不匹配 (2001)

### 3. 存在性错误 (3000-3999)
- **[FileExistError](exist/README.md)**：文件已存在 (3001)
- **[VarExistError](exist/README.md)**：变量/列已存在 (3002)

## 错误代码参考

| 错误代码 | 错误类型 | 描述 |
|----------|----------|------|
| 1001 | FileNotFoundError | 文件不存在 |
| 1002 | VarNotFoundError | 变量/列未找到 |
| 1003 | ConditionNotFoundError | 缺少条件表达式 |
| 1004 | OperatorNotFoundError | 无效或缺少运算符 |
| 2001 | LengthNotMatchError | 数组维度不匹配 |
| 3001 | FileExistError | 文件已存在 |
| 3002 | VarExistError | 变量/列已存在 |

## 快速示例

```python
from deepecon.core.errors import (
    VarNotFoundError,
    LengthNotMatchError,
    FileNotFoundError
)

try:
    # 您的数据处理代码
    result = some_transform(df)
except VarNotFoundError as e:
    print(f"变量错误: {e}")
except LengthNotMatchError as e:
    print(f"维度错误: {e}")
except FileNotFoundError as e:
    print(f"文件错误: {e}")
```

## 文档链接

每种错误都提供直接链接到详细文档：
- **未找到错误**：[文档](not_found/README.md)
- **长度错误**：[文档](length/README.md)
- **存在性错误**：[文档](exist/README.md)

## 使用方法

所有错误都继承自 `ErrorBase`，提供：
- 自动日志记录
- 文档链接
- 错误代码
- 网页浏览器集成