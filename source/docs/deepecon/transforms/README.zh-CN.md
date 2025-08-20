# DeepEcon 数据转换模块
- [English](README.md)
- [简体中文](README.zh-CN.md)

用于计量经济分析的数据转换和预处理工具。

## 概述

转换模块提供了一套全面的数据转换工具，专为计量经济分析而设计。这些转换包括数学运算、数据清洗、相关性分析和统计转换。

## 模块结构

```
deepcon/transforms/
├── __init__.py          # 公共API导出
├── basic/               # 基础数学运算
│   ├── __init__.py
│   ├── base.py         # 数学转换基类
│   └── summarize.py    # 统计汇总
├── corr/               # 相关性分析
│   ├── __init__.py
│   ├── base.py         # 相关性基类
│   ├── pearson.py      # 皮尔逊相关
│   ├── spearman.py     # 斯皮尔曼等级相关
│   ├── kendall.py      # 肯德尔tau相关
│   ├── cramer_v.py     # 克莱姆V相关
│   ├── phi.py          # phi系数
│   ├── distance.py     # 距离相关
│   └── point_biserial.py # 点二列相关
├── drop.py             # 数据过滤和选择
└── winsor2.py          # 异常值处理的缩尾处理
```

## 可用转换

### 1. 基础运算 (`basic`)

#### BasicMath
数据列数学转换的基类。

**特性：**
- 支持自定义数学表达式
- 列创建和替换
- 条件执行支持

#### Summarize
为数据列生成综合统计摘要。

**统计指标：**
- 变量名 (Var)
- 样本量 (N)
- 均值、标准差
- 最小值、最大值
- 四分位数 (Q1, Q3)
- 中位数
- 缺失值计数
- 唯一值计数

**使用示例：**
```python
from deepecon.transforms import Summarize

summarizer = Summarize(df)
summary = summarizer.transform(
    X_cols=['收入', '年龄', '教育'],
    summ_cols=['Var', 'N', 'Mean', 'Std', 'Min', 'Max']
)
```

### 2. 相关性分析 (`corr`)

#### PearsonCorr
计算变量间的皮尔逊相关系数。

**方法：**
- **数组方法**：多个变量的相关矩阵
- **y-x方法**：一个因变量与多个自变量的相关

**使用示例：**
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
    X_cols=['教育', '经验', '年龄']
)
```

### 3. 数据过滤 (`drop`)

#### DropVar
从数据集中删除指定的列。

**参数：**
- `X_cols`：要删除的列名列表
- `_if_exp`：用于过滤的条件表达式

#### KeepVar
仅保留指定的列，删除其他所有列。

**参数：**
- `X_cols`：要保留的列名列表
- `_if_exp`：用于过滤的条件表达式

#### DropCondition
基于条件删除行。

**参数：**
- `_if_exp`：用于行过滤的条件表达式

#### KeepCondition
仅保留满足条件的行。

**参数：**
- `_if_exp`：用于行过滤的条件表达式

**使用示例：**
```python
from deepecon.transforms import DropCondition, KeepVar
from deepecon.core.condition import Condition

# 删除年龄小于18的行
df_filtered = DropCondition(df).transform(
    _if_exp=Condition(lambda x: x['年龄'] < 18)
)

# 仅保留收入和教育列
df_subset = KeepVar(df).transform(X_cols=['收入', '教育'])
```

### 4. 异常值处理 (`winsor2`)

#### Winsor2
通过将极值设置为指定百分位数来进行缩尾处理。

**参数：**
- `p`：上下百分位数的元组（默认：(0.01, 0.99)）
- `suffix`：新列名的后缀（默认：'_w'）
- `replace`：是否替换原始列

**使用示例：**
```python
from deepecon.transforms import Winsor2

# 对收入和年龄变量进行缩尾处理
df_winsorized = Winsor2(df).transform(
    X_cols=['收入', '年龄'],
    p=(0.05, 0.95),
    suffix='_缩尾',
    replace=False
)
```

## API参考

### 公共导出

可直接从 `deepecon.transforms` 获取：

- `BasicMath`：基础数学运算
- `Summarize`：统计汇总
- `PearsonCorr`：皮尔逊相关分析
- `DropVar`：列删除
- `KeepVar`：列选择
- `DropCondition`：按条件行过滤
- `KeepCondition`：按条件行选择

### 高级用法

所有转换都继承自 `TransformBase` 并支持：

- **条件执行** 通过 `_if_exp` 参数
- **列选择** 通过 `X_cols` 参数
- **原地修改** 通过 `replace` 参数
- **方法链式调用** 用于复杂数据管道

## 最佳实践

1. **数据验证**：转换前始终验证输入数据
2. **列选择**：使用显式列列表以获得更好性能
3. **条件逻辑**：利用条件进行目标转换
4. **内存管理**：使用 `replace=True` 进行原地操作以节省内存
5. **管道设计**：链接转换以处理复杂预处理工作流

## 错误处理

转换模块提供全面的错误处理：

- `VarNotFoundError`：指定列不存在时
- `ConditionNotFoundError`：缺少必需条件时
- `LengthNotMatchError`：数据维度不兼容时
- `OperatorNotFoundError`：数学运算符无效时

## 示例

### 完整数据处理管道

```python
from deepecon.transforms import *
from deepecon.core.condition import Condition

# 创建处理管道
df_processed = (
    Winsor2(df)
    .transform(X_cols=['收入', '年龄'], p=(0.01, 0.99))
)

df_processed = (
    DropCondition(df_processed)
    .transform(_if_exp=Condition(lambda x: x['年龄'] >= 18))
)

# 生成统计摘要
summary = Summarize(df_processed).transform(X_cols=['收入', '年龄'])

# 计算相关性
correlations = PearsonCorr(df_processed).transform(
    y_col='收入',
    X_cols=['年龄', '教育', '经验']
)
```