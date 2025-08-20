# DeepEcon/transforms

- [English](README.md)
- [简体中文](README.zh-CN.md)

用于计量经济分析的数据转换和预处理模块。

## 模块结构

- **[basic](basic/README.md)**: 基础数学运算和统计摘要
- **[corr](corr/README.md)**: 相关性分析工具  
- **[drop](drop/README.md)**: 数据过滤和选择
- **[winsor2](winsor2/README.md)**: 通过缩尾处理异常值

## 可用转换

| 模块 | 类 | 用途 |
|------|---|------|
| `basic` | BasicMath, Summarize | 数学运算和统计 |
| `corr` | PearsonCorr | 相关性分析 |
| `drop` | DropVar, KeepVar, DropCondition, KeepCondition | 数据过滤 |
| `winsor2` | Winsor2 | 异常值处理 |

## 快速使用

```python
from deepecon.transforms import Summarize, PearsonCorr

# 基础统计摘要
summary = Summarize(df).transform(X_cols=['收入', '年龄'])

# 相关性分析
corrs = PearsonCorr(df).transform(y_col='收入', X_cols=['年龄', '教育'])
```

## API参考

所有转换继承自`TransformBase`，提供：
- 条件执行通过`_if_exp`
- 列选择通过`X_cols`
- 原地修改通过`replace`
- 特定错误代码的错误处理