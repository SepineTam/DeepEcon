#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : point_biserial.py

import pandas as pd
from scipy.stats import pointbiserialr

from .base import CorrelationBase


class PointBiserialCorr(CorrelationBase):
    """Point-biserial correlation coefficient for binary and continuous variables."""

    def _base_corr(self,
                   a_col: str,
                   b_col: str,
                   *args, **kwargs) -> float:
        """
        Calculate point-biserial correlation coefficient.

        Args:
            a_col: Name of the first column (binary variable)
            b_col: Name of the second column (continuous variable)

        Returns:
            Point-biserial correlation coefficient
        """
        x = pd.to_numeric(self.df[a_col], errors="coerce")
        y = pd.to_numeric(self.df[b_col], errors="coerce")

        # Remove NaN values
        mask = ~(x.isna() | y.isna())
        x_clean = x[mask]
        y_clean = y[mask]

        if len(x_clean) < 2:
            return float("nan")

        # Check if one of the variables is binary
        unique_x = x_clean.nunique()
        unique_y = y_clean.nunique()

        if unique_x == 2 or unique_y == 2:
            # Ensure x is the binary variable
            if unique_y == 2:
                x_clean, y_clean = y_clean, x_clean

            corr, _ = pointbiserialr(x_clean, y_clean)
            return float(corr)
        else:
            # If neither variable is binary, return NaN
            return float("nan")
