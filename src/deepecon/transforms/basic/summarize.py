#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : summarize.py

from typing import Callable, Dict, List, Optional

import pandas as pd

from ...core.base import TransformBase
from ...core.condition import Condition


class Summarize(TransformBase):
    name = "summarize"

    def _mapping_sum_key(self) -> Dict[str, Callable[[pd.Series], str]]:
        """Return a mapping of summary statistics keys to their corresponding functions."""
        sum_mapping: Dict[str, Callable[[pd.Series], str]] = {
            "Var": lambda x: str(x.name),
            "N": lambda x: str(len(x.dropna())),
            "Mean": lambda x: "{:.4f}".format(x.mean()) if pd.api.types.is_numeric_dtype(x) else "-",
            "Std": lambda x: "{:.4f}".format(x.std()) if pd.api.types.is_numeric_dtype(x) else "-",
            "Min": lambda x: "{:.4f}".format(x.min()) if pd.api.types.is_numeric_dtype(x) else "-",
            "Max": lambda x: "{:.4f}".format(x.max()) if pd.api.types.is_numeric_dtype(x) else "-",
            "Q1": lambda x: "{:.4f}".format(x.quantile(0.25)) if pd.api.types.is_numeric_dtype(x) else "-",
            "Q3": lambda x: "{:.4f}".format(x.quantile(0.75)) if pd.api.types.is_numeric_dtype(x) else "-",
            "Median": lambda x: "{:.4f}".format(x.median()) if pd.api.types.is_numeric_dtype(x) else "-",
            "Missing": lambda x: str(x.isna().sum()),
            "Unique": lambda x: str(x.nunique()),
        }
        return sum_mapping

    def options(self) -> Dict[str, str]:
        return self.std_ops(
            ["X_cols", "_if_exp"],
            add_ops={
                "summ_cols": "List of summary statistics to compute."
            }
        )

    def transform(self,
                  y_col: Optional[str] = None,
                  X_cols: Optional[List[str]] = None,
                  _if_exp: Optional[Condition] = None,
                  replace: bool = False,
                  summ_cols: Optional[List[str]] = None,
                  *args, **kwargs) -> pd.DataFrame:
        """Generate summary statistics for specified columns.

        Args:
            X_cols: List of column names to summarize. If None, uses all columns
            _if_exp: Condition expression to filter rows before summarizing
            summ_cols: List of summary statistics to compute
            *args, **kwargs: Additional arguments passed to parent methods

        Returns:
            DataFrame containing summary statistics with columns as requested statistics
            and rows corresponding to the original columns
        """
        # Handle X_cols being None by using all columns
        columns_to_process = X_cols if X_cols is not None else list(self.df.columns)
        self.pre_process(columns_to_process, _if_exp, is_dropna=False)

        if summ_cols is None:
            summ_cols = ["Var", "N", "Mean", "Std", "Min", "Max", "Q1", "Q3", "Missing"]

        # Validate summ_cols against available summary functions
        sum_mapping = self._mapping_sum_key()
        valid_summ_cols = [col for col in summ_cols if col in sum_mapping]

        if not valid_summ_cols:
            raise ValueError("No valid summary columns specified")

        # Generate summary for each column
        summary_list = []
        for col in columns_to_process:
            col_summary = self._summrize_col(col)
            summary_list.append(col_summary[valid_summ_cols])

        # Combine all summaries into a dataframe
        result = pd.DataFrame(summary_list)
        result.index = columns_to_process

        return result

    def _summrize_col(self, col: str) -> pd.Series:
        """Generate summary statistics for a single column."""
        sum_mapping = self._mapping_sum_key()
        series = self.df[col]

        summary_data = {}
        for key, func in sum_mapping.items():
            try:
                summary_data[key] = func(series)
            except (TypeError, ValueError):
                summary_data[key] = "-"

        return pd.Series(summary_data, name=col)
