#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : drop.py

from typing import Dict, List, Optional

import pandas as pd

from ..core.base import TransformBase
from ..core.condition import Condition
from ..core.errors import ConditionNotFoundError, VarNotFoundError


class _BaseVar(TransformBase):
    def options(self) -> Dict[str, str]:
        return self.std_ops(["X_cols"])


class _BaseCondition(TransformBase):
    def options(self) -> Dict[str, str]:
        return self.std_ops(["_if_exp"])


class DropVar(_BaseVar):
    name: str = "drop_var"

    def transform(
        self,
        y_col: Optional[str] = None,
        X_cols: Optional[List[str]] = None,
        _if_exp: Optional[Condition] = None,
        replace: bool = False,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        if not X_cols:
            raise ValueError("Missing the vars to drop")
        if not set(X_cols).issubset(set(self.df.columns)):
            raise VarNotFoundError(X_cols)

        self.df = self.df.drop(columns=X_cols)
        return self.df


class KeepVar(_BaseVar):
    name: str = "keep_var"

    def transform(
        self,
        y_col: Optional[str] = None,
        X_cols: Optional[List[str]] = None,
        _if_exp: Optional[Condition] = None,
        replace: bool = False,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        if not X_cols:
            raise ValueError("Missing the vars to keep")
        if not set(X_cols).issubset(set(self.df.columns)):
            raise VarNotFoundError(X_cols)
        self.df = self.df[X_cols]
        return self.df


class DropCondition(_BaseCondition):
    name: str = "drop_condition"

    def transform(
        self,
        y_col: Optional[str] = None,
        X_cols: Optional[List[str]] = None,
        _if_exp: Optional[Condition] = None,
        replace: bool = False,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        if _if_exp is None:
            raise ConditionNotFoundError("Missing the condition of drop")

        mask = _if_exp(self.df)
        self.df = self.df[~mask]

        return self.df


class KeepCondition(_BaseCondition):
    name: str = "keep_condition"

    def transform(
        self,
        y_col: Optional[str] = None,
        X_cols: Optional[List[str]] = None,
        _if_exp: Optional[Condition] = None,
        replace: bool = False,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        if _if_exp is None:
            raise ConditionNotFoundError("Missing the condition of keep")

        mask = _if_exp(self.df)
        self.df = self.df[mask]

        return self.df
