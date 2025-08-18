# Dev Plan

## Overview (The achieved item will remove after 2 version)
| name                    | description                                  | belong          | version | state |
|-------------------------|----------------------------------------------|-----------------|---------|-------|
| DEconResultMthd         | DeepEcon format ResultMthd into core.results | core.results    | v0.1.2  | - []  |
| DiD FrameWork           | Difference-in-Difference FrameWork           | estimators.dml  | v0.1.3  | - [ ] |
| args: weight            | support run a regression with weight option  | estimators.base | v0.1.4  | - [ ] |
| ResultBase              | The format of output                         | base            | v0.1.2  | - [ ] |
| DML FrameWork           | Double Machine Learning FrameWork            | estimators.dml  | v0.1.3  | - [ ] |
| Other correlation       | Other correlation coefficient                | transforms.corr | v0.1.2  | - [ ] |
| BySort()                | Support run sth with by(by_col: str) option  | core.by         | v0.1.3  | - [ ] |
| correlation coefficient | PearsonCorr correlation coefficient          | transforms      | v0.1.1  | - [x] |
| OLS                     | Ordinary Least Squares                       | estimators      | v0.1.1  | - [x] |
| StataResult             | The output result format                     | base            | v0.1.1  | - [x] |
| winsor2                 | winsor function                              | transforms      | v0.1.1  | - [x] |

## v0.1.1 (Achieved)
- [x] Add OLS into estimators
- [x] Add winsor2 function to transforms
- [x] 20250818 | Add PearsonCorr correlation coefficient into transforms
- [x] Add StataResult into base

## v0.1.2
- [ ] Update ResultBase FrameWork
- [ ] Add Other correlation coefficient
- [ ] Add DeepEcon format ResultMthd into core.results

## v0.1.3
- [ ] Add DML FrameWork
- [ ] Add BySort option
- [ ] Add DiD FrameWork

## v0.1.4
- [ ] Add weight option to OLS (Estimators.base)

