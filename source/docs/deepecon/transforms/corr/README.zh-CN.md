# DeepEcon/transforms/corr

- [English](README.md)
- [简体中文](README.zh-CN.md)

用于测量变量间关系的相关性分析工具。

## 类

### PearsonCorr
计算变量间的皮尔逊相关系数。

**方法：**
- **数组方法**: 多个变量的相关矩阵
- **y-x方法**: 因变量和自变量间的相关

## 使用

```python
from deepecon.transforms import PearsonCorr

# 相关矩阵
corr_matrix = PearsonCorr(df).transform(
    X_cols=['变量1', '变量2', '变量3'],
    is_array=True
)

# y与x的相关
correlations = PearsonCorr(df).transform(
    y_col='收入',
    X_cols=['年龄', '教育', '经验']
)
```

## 参数

| 参数 | 类型 | 描述 |
|-----------|------|-------------|
| `y_col` | str | 目标变量(用于y-x方法) |
| `X_cols` | List[str] | 自变量 |
| `is_array` | bool | 是否计算相关矩阵 |