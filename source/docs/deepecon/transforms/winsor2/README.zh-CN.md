# DeepEcon/transforms/winsor2

- [English](README.md)
- [简体中文](README.zh-CN.md)

通过将极值设置为指定百分位数来进行缩尾处理以处理异常值。

## 类

### Winsor2
通过将极值限制在指定百分位数来进行缩尾处理以处理异常值。

**特性：**
- 可配置的百分位边界
- 可选的列替换
- 新列的自定义后缀
- 条件执行支持

## 使用

```python
from deepecon.transforms import Winsor2

# 基础缩尾处理
df_winsorized = Winsor2(df).transform(
    X_cols=['收入', '年龄'],
    p=(0.01, 0.99)
)

# 使用自定义后缀
df_winsorized = Winsor2(df).transform(
    X_cols=['收入'],
    p=(0.05, 0.95),
    suffix='_缩尾',
    replace=False
)
```

## 参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `X_cols` | List[str] | - | 要缩尾处理的列 |
| `p` | Tuple[float, float] | (0.01, 0.99) | 上下百分位数 |
| `suffix` | str | "_w" | 新列名的后缀 |
| `replace` | bool | False | 是否替换原始列 |