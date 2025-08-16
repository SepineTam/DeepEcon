#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : base.py

from __future__ import annotations

import calendar
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

import pandas as pd

from .condition import Condition
from .errors import VarNotFoundError
from .results import ResultStrMthdBase, get_render, list_renderers


@dataclass
class DataFrameBase(ABC):
    def __init__(self):
        pass


@dataclass
class ResultBase(ABC):
    data: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    ts: datetime = field(default_factory=datetime.now)
    mthd: Literal[list_renderers()] = field(default="stata")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "mthd" in kwargs and isinstance(kwargs["mthd"], str):
            self.mthd = kwargs["mthd"]

    def __post_init__(self):
        self.meta["Date"] = self.ts.date()
        self.meta["time"] = self.ts.time()
        self.meta["weekday"] = calendar.day_name[self.ts.weekday()]

    def _update_meta(self, key: str, value: Any) -> None:
        self.meta[key] = value

    def _update_data(self, key: str, value: Any) -> None:
        self.data[key] = value

    @abstractmethod
    def meta_keys(self) -> List[str]: ...

    @abstractmethod
    def data_keys(self) -> List[str]: ...

    def __str__(self):
        mthd_class: ResultStrMthdBase = get_render(self.mthd)
        return mthd_class.render(self)


class Base(ABC):
    name: str = "function name"
    _std_ops = {
        "X_cols": "The columns name of X position",
        "y_col": "The column name of y position",
        "replace": "Whether replace the previous col with the newer one",
        "_if_exp": "The exception condition",
        "weight": "The weight of each observation",
    }

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.ops = self.options()

    @abstractmethod
    def __call__(self, *args, **kwargs) -> pd.DataFrame: ...

    @abstractmethod
    def options(self) -> Dict[str, str]: ...

    """The args supported for this class conducted with dict"""

    def std_ops(
        self, keys: Optional[List[str]] = None, add_ops: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
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

    def check_var_exists(self, var_name: str) -> bool:
        """
        Check if a variable (column) exists in the DataFrame.

        Args:
            var_name (str): The name of the variable/column to check.

        Returns:
            bool: True if the variable exists in the DataFrame, False otherwise.
        """
        return var_name in self.df.columns

    def check_vars_exist(self, var_names: List[str]) -> List[bool]:
        """
        Check if multiple variables (columns) exist in the DataFrame.

        Args:
            var_names (List[str]): List of variable/column names to check.

        Returns:
            List[bool]: List of boolean values indicating whether each
                       variable exists in the DataFrame.
        """
        return [var_name in self.df.columns for var_name in var_names]


class EstimatorBase(Base):
    name: str = "estimator"

    def __call__(self, *args, **kwargs) -> pd.DataFrame:
        return self.estimator(*args, **kwargs)

    @abstractmethod
    def estimator(
        self,
        y_cols: str,
        X_cols: List[str],
        _if_exp: Optional[Condition] = None,
        weight: Optional[pd.Series | str] = None,
        *args,
        **kwargs,
    ) -> pd.DataFrame: ...

    def pre_process(
        self, y_cols: str, X_cols: List[str], _if_exp: Optional[Condition] = None
    ) -> None:
        target_columns: List[str] = [y_cols] + X_cols
        if not all(self.check_vars_exist(target_columns)):
            raise VarNotFoundError(y_cols, X_cols)
        self.df = self.df.dropna(subset=target_columns)
        self._condition_on(_if_exp)

    def _condition_on(self, _if_exp: Optional[Condition] = None) -> pd.DataFrame:
        """Filter the DataFrame based on a condition and modify it in-place.

        This method applies the given condition to filter the DataFrame. If a condition
        is provided, it modifies self.df in-place to contain only rows that satisfy
        the condition. If no condition is provided, returns the DataFrame unchanged.

        Args:
            _if_exp (Optional[Condition]):
                The condition object used to filter the DataFrame.
                If None, no filtering is applied. Defaults to None.

        Returns:
            pd.DataFrame: The filtered DataFrame. Note that this is a reference to
                the modified self.df, not a copy.
        """
        if _if_exp is None:
            return self.df
        else:
            mask = _if_exp.to_mask(self.df)
            self.df = self.df[mask]
        return self.df


class TransformBase(Base):
    name: str = "transform"  # must be unique

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    @abstractmethod
    def transform(
        self,
        y_cols: Optional[List[str]] = None,
        X_cols: Optional[List[str]] = None,
        _if_exp: Optional[Condition] = None,
        replace: bool = False,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
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
