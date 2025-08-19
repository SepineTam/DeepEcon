#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : kendall.py

import pandas as pd
from scipy.stats import kendalltau

from .base import CorrelationBase


class KendallCorr(CorrelationBase):
    """Kendall's Tau rank correlation coefficient."""

    def _base_corr(self,
                   a_col: str,
                   b_col: str,
                   *args, **kwargs) -> float:
        """
        Calculate Kendall's Tau correlation coefficient between two columns.

        Args:
            a_col: Name of the first column
            b_col: Name of the second column

        Returns:
            Kendall's Tau correlation coefficient
        """
        x = pd.to_numeric(self.df[a_col], errors="coerce")
        y = pd.to_numeric(self.df[b_col], errors="coerce")

        # Remove NaN values
        mask = ~(x.isna() | y.isna())
        x_clean = x[mask]
        y_clean = y[mask]

        if len(x_clean) < 2:
            return float("nan")

        tau, _ = kendalltau(x_clean, y_clean)
        return float(tau)
