#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : base.py

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd

from ...core.base import TransformBase
from ...core.condition import Condition
from ...core.errors import OperatorNotFoundError


class BasicMathBase(TransformBase, ABC):
    def options(self) -> Dict[str, str]:
        return self.std_ops(
            ["y_col"], add_ops={"op": "the operator of mathematical expressions"}
        )

    def transform(
        self,
        y_col: Optional[str] = None,
        X_cols: Optional[List[str]] = None,
        _if_exp: Optional[Condition] = None,
        replace: bool = False,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        # check for the required arguments
        if not y_col:
            raise TypeError("Missing 1 required positional argument: 'y_col'")
        if not isinstance(y_col, str):
            raise TypeError("'y_col' is not a string")

        op = kwargs.get("op", None)
        if not op:
            raise TypeError("Missing 1 required positional argument: 'op'")
        if not isinstance(op, str):
            raise OperatorNotFoundError("'op' is not a string")

        # set the target column
        target_col = y_col

        # find whether 'target_col' is in the dataframe
        if y_col in self.df.columns and not replace:
            raise ValueError(
                f"Column '{target_col}' already exists, please turn 'replace' on True"
            )

        result = self._evaluate_expression(op)
        self.df[target_col] = result
        return self.df

    @abstractmethod
    def _evaluate_expression(self, expr) -> pd.Series: ...
