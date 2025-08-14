#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : drop.py

from typing import List, Optional

import pandas as pd

from ..core.base import TransformBase
from ..core.condition import Condition
from ..core.errors import ConditionNotFoundError, VarNotFoundError


class DropVar(TransformBase):
    name: str = "drop_var"

    def __call__(self,
                 X_cols: Optional[List[str]] = None,
                 *args, **kwargs) -> pd.DataFrame:
        if not X_cols:
            raise ValueError("Missing the vars to drop")
        if not set(X_cols).issubset(set(self.df.columns)):
            raise VarNotFoundError(X_cols)

        self.df = self.df.drop(columns=X_cols)
        return self.df


class KeepVar(TransformBase):
    name: str = "keep_var"

    def __call__(self,
                 X_cols: Optional[List[str]] = None,
                 *args, **kwargs) -> pd.DataFrame:
        if not X_cols:
            raise ValueError("Missing the vars to keep")
        if not set(X_cols).issubset(set(self.df.columns)):
            raise VarNotFoundError(X_cols)
        self.df = self.df[X_cols]
        return self.df


class DropCondition(TransformBase):
    name: str = "drop_condition"

    def __call__(self,
                 _if_exp: Optional[Condition] = None,
                 *args, **kwargs) -> pd.DataFrame:
        if _if_exp is None:
            raise ConditionNotFoundError("Missing the condition of drop")

        mask = _if_exp(self.df)
        self.df = self.df[~mask]

        return self.df


class KeepCondition(TransformBase):
    name: str = "keep_condition"

    def __call__(self,
                 _if_exp: Optional[Condition] = None,
                 *args, **kwargs) -> pd.DataFrame:
        if _if_exp is None:
            raise ConditionNotFoundError("Missing the condition of keep")

        mask = _if_exp(self.df)
        self.df = self.df[mask]

        return self.df
