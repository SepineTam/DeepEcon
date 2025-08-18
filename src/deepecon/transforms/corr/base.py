#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : base.py
from abc import abstractmethod
from typing import Dict, List, Optional

import pandas as pd

from ...core.base import TransformBase
from ...core.condition import Condition
from ...core.errors import LengthNotMatchError


class CorrelationBase(TransformBase):
    name = "correlation"

    def options(self) -> Dict[str, str]:
        return self.std_ops(["X_cols"])

    def transform(self,
                  y_col: Optional[str] = None,
                  X_cols: Optional[List[str]] = None,
                  _if_exp: Optional[Condition] = None,
                  replace: bool = False,
                  *args, **kwargs) -> pd.DataFrame:
        is_array = kwargs.get("is_array", False)
        if is_array:
            if len(X_cols) < 2:
                raise LengthNotMatchError("With array method, "
                                          "X_cols length must over 2")
            self.pre_process(X_cols, _if_exp)
            R = self.array_corr(X_cols)
        else:
            if not isinstance(y_col, str):
                raise TypeError("y_col must be a string")
            if len(X_cols) == 0:
                raise LengthNotMatchError("With y_x method, "
                                          "X_cols length must over 0")
            target_columns: List[str] = [y_col] + X_cols
            self.pre_process(target_columns, _if_exp)
            R = self.y_x_corr(y_col, X_cols)
        return R

    def array_corr(self,
                   X_cols: List[str]) -> pd.DataFrame:
        R = pd.DataFrame(index=X_cols, columns=X_cols)
        for col_i in X_cols:
            for col_j in X_cols:
                if col_i > col_j:
                    r = self._base_corr(col_i, col_j)
                    R[col_i][col_j] = r
                    R[col_j][col_i] = r
                elif col_i == col_j:
                    R[col_i][col_j] = float(1)
                else:
                    pass
        return R

    def y_x_corr(self,
                 y_col: str,
                 X_cols: List[str]) -> pd.DataFrame:
        R_dict: Dict[str, float] = {}
        for x_col in X_cols:
            R_dict[x_col] = self._base_corr(y_col, x_col)
        R = pd.DataFrame([R_dict])
        return R

    @abstractmethod
    def _base_corr(self,
                   a_col: str,
                   b_col: str,
                   *args, **kwargs) -> float:
        ...
