# DeepEcon/errors/exist

- [English](README.md)
- [简体中文](README.zh-CN.md)

当尝试创建或访问已存在内容时的错误类型。

## 错误类型

- **FileExistError (3001)**: 文件已存在
- **VarExistError (3002)**: 变量/列已存在

## 快速参考

| 错误代码 | 描述 | 示例修复 |
|----------|------|----------|
| 3001 | 创建文件前检查 | `os.path.exists(filepath)` |
| 3002 | 使用replace=True参数 | `transform(..., replace=True)` |

## 使用方法

```python
from deepecon.core.errors import VarExistError

try:
    # 尝试创建已存在的列
    pass
except VarExistError as e:
    print(f"变量已存在: {e}")
    # 通过使用replace=True或选择不同名称来处理
```