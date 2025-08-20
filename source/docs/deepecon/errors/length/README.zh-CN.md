# DeepEcon/errors/length

- [English](README.md)
- [简体中文](README.zh-CN.md)

当数据维度不符合预期要求时的错误类型。

## 错误类型

- **LengthNotMatchError (2001)**: 数组或维度不匹配

## 快速参考

| 错误代码 | 描述 | 示例修复 |
|----------|------|----------|
| 2001 | 确保兼容维度 | 相关性分析 `len(X_cols) >= 2` |

## 使用方法

```python
from deepecon.core.errors import LengthNotMatchError

try:
    # 尝试维度不匹配的操作
    pass
except LengthNotMatchError as e:
    print(f"维度不匹配: {e}")
    # 确保数组有兼容长度
```

## 常见情况

- **相关矩阵**: 需要至少2个变量
- **数组运算**: 需要匹配维度
- **统计函数**: 需要最小数据点