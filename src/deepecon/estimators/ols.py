#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : ols.py

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..core.base import EstimatorBase
from ..core.condition import Condition
from ._out import EstimatorResult


class OrdinaryLeastSquares(EstimatorBase):
    name = "ols"

    def options(self) -> Dict[str, str]:
        return self.std_ops(
            keys=["y_col", "X_cols"],
            add_ops={
                "is_cons": "Whether to add a constant column to the design matrix",
            }
        )

    def estimator(self,
                  y_col: str,
                  X_cols: List[str],
                  _if_exp: Optional[Condition] = None,
                  weight: Optional[str] = None,
                  is_cons: bool = True,
                  *args, **kwargs) -> EstimatorResult:
        # make sure all the args are exist and prepare data
        target_columns: List[str] = [y_col] + X_cols
        self.pre_process(target_columns, _if_exp)
        y_data = self.df[y_col].to_numpy(dtype=float)
        X_data = self.df[X_cols].to_numpy(dtype=float)
        if is_cons:
            X_data = np.hstack([np.ones((X_data.shape[0], 1)), X_data])
        # n is the number of samples, k is the number of independent variables
        n, k = X_data.shape

        beta_hat = np.linalg.inv(X_data.T @ X_data) @ X_data.T @ y_data
        y_hat = X_data @ beta_hat
        residuals = y_data - y_hat

        pass
