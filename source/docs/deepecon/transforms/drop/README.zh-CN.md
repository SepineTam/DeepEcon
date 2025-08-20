# DeepEcon/transforms/drop

- [English](README.md)
- [简体中文](README.zh-CN.md)

用于管理数据集列和行的数据过滤和选择工具。

## 类

### DropVar
从数据集中删除指定列。

### KeepVar
仅保留指定列，删除其他所有列。

### DropCondition
基于条件表达式删除行。

### KeepCondition
仅保留满足条件的行。

## 使用

```python
from deepecon.transforms import DropVar, DropCondition
from deepecon.core.condition import Condition

# 删除列
df_clean = DropVar(df).transform(X_cols=['临时列'])

# 过滤行
df_filtered = DropCondition(df).transform(
    _if_exp=Condition(lambda x: x['年龄'] >= 18)
)
```

## 参数

| 类 | 关键参数 | 描述 |
|-------|----------------|-------------|
| DropVar | `X_cols` | 要删除的列列表 |
| KeepVar | `X_cols` | 要保留的列列表 |
| DropCondition | `_if_exp` | 过滤行的条件 |
| KeepCondition | `_if_exp` | 保留行的条件 |