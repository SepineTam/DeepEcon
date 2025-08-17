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
from ..core.errors import VarNotFoundError


class OrdinaryLeastSquares(EstimatorBase):
    name = "ols"

    def options(self) -> Dict[str, str]:
        return self.std_ops(["y_col", "X_cols"])

    def estimator(self,
                  y_col: str,
                  X_cols: List[str],
                  _if_exp: Optional[Condition] = None,
                  is_cons: Optional[str] = None,
                  *args, **kwargs) -> pd.DataFrame:
        # make sure all the args are exist and prepare data
        self.pre_process(y_col, X_cols, _if_exp)
        y_data = self.df[y_col].to_numpy(dtype=float)
        X_data = self.df[X_cols].to_numpy(dtype=float)

        # main algorithm
        pass
