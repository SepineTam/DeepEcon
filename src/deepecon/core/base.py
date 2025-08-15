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
from typing import Dict, List, Optional

import pandas as pd

from .condition import Condition


@dataclass
class DataFrameBase(ABC):
    def __init__(self): pass


class Base(ABC):
    name: str = "function name"
    _std_ops = {
        "X_cols": "The columns name of X position",
        "y_cols": "The columns name of y position",
        "replace": "Whether replace the previous col with the newer one",
        "_if_exp": "The exception condition",
        "weight": "The weight of each observation"
    }

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.ops = self.options()

    @abstractmethod
    def __call__(self, *args, **kwargs) -> pd.DataFrame:
        ...

    @abstractmethod
    def options(self) -> Dict[str, str]:
        ...

    """The args supported for this class conducted with dict"""

    def std_ops(self,
                keys: Optional[List[str]] = None,
                add_ops: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Get function standard operation descriptions.

        Args:
           keys: Optional list of keys to filter the standard operations. If None,
               all standard operations are included. Only keys that exist in the
               standard operations dictionary will be returned.
           add_ops: Optional dictionary of additional operations to include in the
               result. These will be merged with the filtered standard operations,
               potentially overriding existing keys.

        Returns:
           A dictionary mapping operation names to their descriptions. Contains
           filtered standard operations and any additional operations provided.

        """
        if keys is None:
            result = self._std_ops.copy()
        else:
            valid_keys = set(keys) & set(self._std_ops.keys())
            result = {key: self._std_ops[key] for key in valid_keys}

        if add_ops is not None:
            result.update(add_ops)

        return result


class ResultBase(Base):
    ...


class EstimatorBase(Base):
    name: str = "estimator"

    def __call__(self, *args, **kwargs) -> pd.DataFrame:
        return self.estimator(*args, **kwargs)

    @abstractmethod
    def estimator(self,
                  y_cols: List[str],
                  X_cols: List[str],
                  _if_exp: Optional[Condition] = None,
                  weight: Optional[pd.Series | str] = None,
                  *args, **kwargs) -> pd.DataFrame: ...


class TransformBase(Base):
    name: str = "transform"  # must be unique

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    @abstractmethod
    def transform(self,
                  y_cols: Optional[List[str]] = None,
                  X_cols: Optional[List[str]] = None,
                  _if_exp: Optional[Condition] = None,
                  replace: bool = False,
                  *args, **kwargs) -> pd.DataFrame:
        """
        Apply transformation to specified columns of the dataframe and return the transformed dataframe.
        The previous df will not be change, you must capture the return value if you want to change it.

        Args:
            (Options) <- if you want to know which options is supported visit self.options
            y_cols (Optional[List[str]]):
                the newer cols name
            X_cols (Optional[List[str]]):
                the cols which need to be processed.
            _if_exp (Optional[Condition)]:
                the exception condition
            replace (bool):
                whether replace the previous col with the newer one.
            ...

        Returns:
            pd.DataFrame: the processed pd.DataFrame
        """
        return self.df
