# DeepEcon
DeepEcon：你的一站式计量经济算法包

[![en](https://img.shields.io/badge/lang-English-red.svg)](../../../../README.md)
[![cn](https://img.shields.io/badge/语言-中文-yellow.svg)](README.md)
[![PyPI version](https://img.shields.io/pypi/v/deepecon.svg)](https://pypi.org/project/deepecon/)
[![PyPI Downloads](https://static.pepy.tech/badge/deepecon)](https://pepy.tech/projects/deepecon)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Issue](https://img.shields.io/badge/Issue-report-green.svg)](https://github.com/sepinetam/deepecon/issues/new)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/SepineTam/DeepEcon)


## 快速开始
### 从 Pypi 安装
```bash
pip install deepecon
```

### 运行回归分析
```python
from deepecon.estimators import OLS
import pandas as pd

df: pd.DataFrame
y_col = 'y'
X_cols = ['x1', 'x2', 'x3']

ols = OLS(df)
result = ols(y_col, X_cols)
```

## 路线图
查看路线图 [这里](../../../../DEVPLAN.md)。

## 许可证
[MIT 许可证](../../../../LICENSE)

## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=sepinetam/deepecon&type=Date)](https://www.star-history.com/#sepinetam/deepecon&Date)

