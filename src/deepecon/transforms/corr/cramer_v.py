#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : cramer_v.py

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from .base import CorrelationBase


class CramerVCorr(CorrelationBase):
    """Cramér's V coefficient for categorical variables."""

    def _base_corr(self,
                   a_col: str,
                   b_col: str,
                   *args, **kwargs) -> float:
        """
        Calculate Cramér's V coefficient for categorical variables.

        Args:
            a_col: Name of the first categorical column
            b_col: Name of the second categorical column

        Returns:
            Cramér's V correlation coefficient
        """
        x = self.df[a_col].astype('category')
        y = self.df[b_col].astype('category')

        # Remove NaN values
        mask = ~(x.isna() | y.isna())
        x_clean = x[mask]
        y_clean = y[mask]

        if len(x_clean) < 2:
            return float("nan")

        # Create contingency table
        contingency = pd.crosstab(x_clean, y_clean)

        # Calculate Cramér's V
        # V = sqrt(chi2 / (n * (min(r-1, c-1))))
        try:
            chi2, _, _, _ = chi2_contingency(contingency)
            n = len(x_clean)
            r, c = contingency.shape

            if n == 0 or min(r-1, c-1) == 0:
                return float("nan")

            v = np.sqrt(chi2 / (n * min(r-1, c-1)))
            return float(v)
        except:
            return float("nan")
