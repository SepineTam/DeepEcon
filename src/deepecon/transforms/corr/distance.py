#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : distance.py

import pandas as pd
from dcor import distance_correlation

from .base import CorrelationBase


class DistanceCorr(CorrelationBase):
    """Distance correlation coefficient for non-linear relationships."""

    def _base_corr(self,
                   a_col: str,
                   b_col: str,
                   *args, **kwargs) -> float:
        """
        Calculate distance correlation coefficient.

        Args:
            a_col: Name of the first column
            b_col: Name of the second column

        Returns:
            Distance correlation coefficient
        """
        try:
            x = pd.to_numeric(self.df[a_col], errors="coerce")
            y = pd.to_numeric(self.df[b_col], errors="coerce")

            # Remove NaN values
            mask = ~(x.isna() | y.isna())
            x_clean = x[mask].values
            y_clean = y[mask].values

            if len(x_clean) < 3:
                return float("nan")

            # Calculate distance correlation
            dcor = distance_correlation(x_clean, y_clean)
            return float(dcor)
        except ImportError:
            # Fallback to NaN if dcor is not available
            return float("nan")
        except:
            return float("nan")
