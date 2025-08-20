# DeepEcon/errors

- [English](README.md)
- [简体中文](README.zh-CN.md)

具有结构化错误代码和文档链接的综合错误处理系统。

## 错误类别

### 未找到错误 (1000-1999)
- **[not_found](not_found/README.md)**: 文件、变量、条件、运算符未找到

### 长度错误 (2000-2999)  
- **[length](length/README.md)**: 数组和维度不匹配

### 存在性错误 (3000-3999)
- **[exist](exist/README.md)**: 已存在的文件和变量

## 错误代码参考

| 代码 | 类型 | 描述 |
|------|------|------|
| 1001 | FileNotFoundError | 文件不存在 |
| 1002 | VarNotFoundError | 变量/列未找到 |
| 1003 | ConditionNotFoundError | 缺少条件表达式 |
| 1004 | OperatorNotFoundError | 无效或缺少运算符 |
| 2001 | LengthNotMatchError | 数组维度不匹配 |
| 3001 | FileExistError | 文件已存在 |
| 3002 | VarExistError | 变量/列已存在 |

## 使用方法

```python
from deepecon.core.errors import VarNotFoundError

try:
    # 您的数据处理代码
    pass
except VarNotFoundError as e:
    print(f"变量未找到: {e}")
```