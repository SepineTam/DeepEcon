#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : base.py

from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd

from ...core.base import TransformBase
from ...core.condition import Condition
from ...core.errors import LengthNotMatchError, OperatorNotFoundError


class BasicMathBase(TransformBase, ABC):
    def __call__(self,
                 X_cols: Optional[List[str]] = None,
                 y_cols: Optional[List[str]] = None,
                 _if_exp: Optional[Condition] = None,
                 replace: bool = False,
                 *args,
                 **kwargs) -> pd.DataFrame:
        # check for the required arguments
        if not y_cols:
            raise TypeError("Missing 1 required positional argument: 'y_clos'")
        if len(y_cols) != 1:
            raise LengthNotMatchError("'y_cols' is too long")

        op = kwargs.get("op", None)
        if not op:
            raise TypeError("Missing 1 required positional argument: 'op'")
        if not isinstance(op, str):
            raise OperatorNotFoundError("'op' is not a string")

        # set the target column
        target_col = y_cols[0]

        # find whether 'target_col' is in the dataframe
        if target_col in self.df.columns and not replace:
            raise ValueError(
                f"Column '{target_col}' already exists, please turn 'replace' on True")

        result = self._evaluate_expression(op)
        self.df[target_col] = result
        return self.df

    @abstractmethod
    def _evaluate_expression(self, expr) -> pd.Series: ...
