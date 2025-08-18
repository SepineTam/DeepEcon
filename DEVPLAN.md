# Dev Plan

## Overview
| name                    | description                                 | belong          | version | state |
|-------------------------|---------------------------------------------|-----------------|---------|-------|
| OLS                     | Ordinary Least Squares                      | estimators      | v0.1.1  | - [x] |
| args: weight            | support run a regression with weight option | estimators.base | v0.1.4  | - [ ] |
| winsor2                 | winsor function                             | transforms      | v0.1.1  | - [ ] |
| ResultBase              | The format of output                        | base            | v0.1.2  | - [ ] |
| DML FrameWork           | Double Machine Learning FrameWork           | estimators.dml  | v0.1.3  | - [ ] |
| correlation coefficient | PearsonCorr correlation coefficient         | transforms      | v0.1.1  | - [x] |

## v0.1.1
- [x] Add OLS into estimators
- [ ] Add winsor2 function to transforms
- [x] 20250818 | Add PearsonCorr correlation coefficient into transforms

## v0.1.2
- [ ] Update ResultBase FrameWork

## v0.1.3
- [ ] Add DML FrameWork

## v0.1.4
- [ ] Add weight option to OLS (Estimators.base)

