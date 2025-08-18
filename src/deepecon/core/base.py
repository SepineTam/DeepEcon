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

import numpy as np
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
    ANOVA: pd.DataFrame = field(default_factory=pd.DataFrame)
    data: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    ts: datetime = field(default_factory=datetime.now)
    mthd: Literal[list_renderers()] = field(default="stata")

    def set_mthd(self, mthd: str):
        if mthd in list_renderers():
            self.mthd = mthd
        else:
            raise ValueError(f"No renderer registered for name {mthd}. "
                             f"Available: {list_renderers()}")

    def __post_init__(self):
        self._anova_init()
        self._meta_init()

    def update_meta(self, key: str, value: Any) -> None:
        self.meta[key] = value

    def update_anova(self, *, key: str, index: str, value: float) -> None:
        anova_indexs: List[str] = self.ANOVA.index
        anova_keys: List[str] = self.ANOVA.columns
        if index not in anova_indexs:
            raise KeyError(f"{index} not found in ANOVA")
        if key not in anova_keys:
            raise KeyError(f"{key} not found in ANOVA")
        self.ANOVA.loc[index, key] = value

    def update_data(self,
                    y_name: str,
                    *,
                    X_names: List[str],
                    beta: np.ndarray,
                    stderr: np.ndarray,
                    t_value: np.ndarray,
                    p_value: np.ndarray,
                    ci_lower: np.ndarray,
                    ci_upper: np.ndarray) -> None:
        self.data["y_name"] = y_name
        self.data["X_names"] = X_names
        self.data["beta"] = beta
        self.data["stderr"] = stderr
        self.data["t_value"] = t_value
        self.data["p_value"] = p_value
        self.data["ci_lower"] = ci_lower
        self.data["ci_upper"] = ci_upper

    @staticmethod
    def __anova_index() -> List[str]:
        return ['Model', 'Residual', 'Total']

    def _meta_init(self) -> List[str]:
        self.meta["Date"] = self.ts.date()
        self.meta["time"] = self.ts.time()
        self.meta["weekday"] = calendar.day_name[self.ts.weekday()]
        for key in self.meta_keys():
            if key not in self.meta:
                self.meta[key] = None

    def _anova_init(self) -> pd.DataFrame:
        idx = pd.Index(self.__anova_index())
        dtypes = {
            'SS': 'float64',
            'df': 'Int64',
            'MS': 'float64'
        }
        self.ANOVA = pd.DataFrame({c: pd.Series(dtype=t) for c, t in dtypes.items()}, index=idx)
        return self.ANOVA

    @abstractmethod
    def meta_keys(self) -> List[str]:
        ...

    def __str__(self):
        mthd_class: ResultStrMthdBase = get_render(self.mthd)
        return mthd_class.render(mthd_class, res=self)


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

    def pre_process(self,
                    target_columns: List[str],
                    _if_exp: Optional[Condition] = None) -> None:
        if not all(self.check_vars_exist(target_columns)):
            raise VarNotFoundError(target_columns, "Not Found")
        self.df = self.df.dropna(subset=target_columns)
        self._condition_on(_if_exp)


class EstimatorBase(Base):
    name: str = "estimator"

    def __call__(self, *args, **kwargs) -> "EstimatorResult":
        return self.estimator(*args, **kwargs)

    @abstractmethod
    def estimator(self,
                  y_col: str,
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
                  y_col: Optional[str] = None,
                  X_cols: Optional[List[str]] = None,
                  _if_exp: Optional[Condition] = None,
                  replace: bool = False,
                  *args, **kwargs) -> pd.DataFrame:
        """
        Apply transformation to specified columns of the dataframe and return the transformed dataframe.
        The previous df will not be change, you must capture the return value if you want to change it.

        Args:
            (Options) <- if you want to know which options is supported visit self.options
            y_col (Optional[str]):
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
