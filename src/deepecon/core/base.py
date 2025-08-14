#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from .condition import Condition


@dataclass
class DataFrameBase(ABC):
    def __init__(self): pass


class ResultBase(ABC):
    def __init__(self): pass


class EstimatorBase(ABC):
    def __init__(self): pass


class TransformBase(ABC):
    name: str = "transform"  # must be unique

    def __init__(self, df: pd.DataFrame):
        self.df = df

    @abstractmethod
    def __call__(self,
                 X_cols: Optional[List[str]] = None,
                 y_cols: Optional[List[str]] = None,
                 _if_exp: Optional[Condition] = None,
                 replace: bool = False,
                 *args,
                 **kwargs) -> pd.DataFrame:
        """
        Apply transformation to specified columns of the dataframe and return the transformed dataframe.
        The previous df will not be change, you must capture the return value if you want to change it.

        Args:
            (Options)
            X_cols (Optional[List[str]]):
                the cols which need to be processed.
            y_cols (Optional[List[str]]):
                the newer cols name
            _if_exp (Optional[Condition)]:
                the exception condition
            replace (bool):
                whether replace the previous col with the newer one.
            ...

        Returns:
            pd.DataFrame: the processed pd.DataFrame
        """
        return self.df
