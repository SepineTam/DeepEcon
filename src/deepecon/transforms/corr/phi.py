#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : phi.py

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from .base import CorrelationBase


class PhiCorr(CorrelationBase):
    """Phi coefficient for binary variables."""

    def _base_corr(self,
                   a_col: str,
                   b_col: str,
                   *args, **kwargs) -> float:
        """
        Calculate Phi coefficient for binary variables.

        Args:
            a_col: Name of the first column (binary variable)
            b_col: Name of the second column (binary variable)

        Returns:
            Phi correlation coefficient
        """
        x = pd.to_numeric(self.df[a_col], errors="coerce")
        y = pd.to_numeric(self.df[b_col], errors="coerce")

        # Remove NaN values
        mask = ~(x.isna() | y.isna())
        x_clean = x[mask]
        y_clean = y[mask]

        if len(x_clean) < 2:
            return float("nan")

        # Check if both variables are binary
        unique_x = x_clean.nunique()
        unique_y = y_clean.nunique()

        if unique_x != 2 or unique_y != 2:
            return float("nan")

        # Create contingency table
        contingency = pd.crosstab(x_clean, y_clean)

        if contingency.shape != (2, 2):
            return float("nan")

        # Calculate Phi coefficient
        # Phi = sqrt(chi2 / n)
        try:
            chi2, _, _, _ = chi2_contingency(contingency)
            n = len(x_clean)
            phi = np.sqrt(chi2 / n)
            return float(phi)
        except Exception as e:
            print(f"Error calculating Phi coefficient: {e}")
            return float("nan")
