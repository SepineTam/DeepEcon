# DeepEcon/errors/not_found

- [English](README.md)
- [简体中文](README.zh-CN.md)

当所需文件、变量、条件或运算符无法找到时的错误类型。

## 错误类型

- **FileNotFoundError (1001)**: 文件不存在
- **VarNotFoundError (1002)**: 变量/列未找到
- **ConditionNotFoundError (1003)**: 缺少条件表达式
- **OperatorNotFoundError (1004)**: 无效或缺少运算符

## 快速参考

| 错误代码 | 描述 | 示例修复 |
|----------|------|----------|
| 1001 | 检查文件路径是否存在 | `os.path.exists(filepath)` |
| 1002 | 验证列名 | `list(df.columns)` |
| 1003 | 提供条件参数 | `_if_exp=Condition(...)` |
| 1004 | 使用有效运算符字符串 | `op='x + 1'` |

## 使用方法

```python
from deepecon.core.errors import VarNotFoundError

try:
    # 你的代码
    pass
except VarNotFoundError as e:
    print(f"变量未找到: {e}")
```