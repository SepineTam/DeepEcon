# DeepEcon
DeepEcon：Your one-stop Python package for econometric algorithms

[![en](https://img.shields.io/badge/lang-English-red.svg)](README.md)
[![cn](https://img.shields.io/badge/语言-中文-yellow.svg)](source/docs/README/cn/README.md)
[![PyPI version](https://img.shields.io/pypi/v/deepecon.svg)](https://pypi.org/project/deepecon/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Issue](https://img.shields.io/badge/Issue-report-green.svg)](https://github.com/sepinetam/deepecon/issues/new)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/SepineTam/DeepEcon)

## Quickly Start
### Install from Pypi
```bash
pip install deepecon
```

### Run a regression
```python
from deepecon.estimators import OLS
import pandas as pd

df: pd.DataFrame
y_col = 'y'
X_cols = ['x1', 'x2', 'x3']

ols = OLS(df)
result = ols(y_col, X_cols)
```

## Roadmap
View the roadmap [here](DEVPLAN.md).

## License
[MIT License](LICENSE)

## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=sepinetam/deepecon&type=Date)](https://www.star-history.com/#sepinetam/deepecon&Date)

