# DeepEcon/transforms/basic

- [English](README.md)
- [简体中文](README.zh-CN.md)

用于数据转换的基础数学运算和统计摘要。

## 类

### BasicMath
数据列数学转换的基类。

**特性：**
- 自定义数学表达式
- 列创建和替换
- 条件执行支持

### Summarize  
为数据列生成综合统计摘要。

**可用统计：**
- 变量名 (Var)
- 样本量 (N)
- 均值、标准差
- 最小值、最大值
- 四分位数 (Q1, Q3)
- 中位数
- 缺失值计数
- 唯一值计数

## 使用

```python
from deepecon.transforms import Summarize

# 生成统计摘要
summary = Summarize(df).transform(
    X_cols=['收入', '年龄'],
    summ_cols=['Var', 'N', 'Mean', 'Std']
)
```

## 参数

| 参数 | 类型 | 描述 |
|-----------|------|-------------|
| `X_cols` | List[str] | 要处理的列 |
| `summ_cols` | List[str] | 要计算的统计量 |