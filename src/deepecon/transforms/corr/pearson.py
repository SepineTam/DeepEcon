#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : pearson.py

from typing import List

import numpy as np
import pandas as pd

from .base import CorrelationBase


class PearsonCorr(CorrelationBase):

    def _base_corr(self,
                   a_col: str,
                   b_col: str,
                   *args, **kwargs) -> float:
        x = pd.to_numeric(self.df[a_col], errors="coerce")
        y = pd.to_numeric(self.df[b_col], errors="coerce")
        s = pd.concat([x, y], axis=1).dropna()

        if s.shape[0] < 2:
            return float("nan")
        x_c = s.iloc[:, 0] - s.iloc[:, 0].mean()
        y_c = s.iloc[:, 1] - s.iloc[:, 1].mean()
        denom = np.sqrt((x_c ** 2).sum() * (y_c ** 2).sum())
        if denom == 0:
            return float("nan")
        return float((x_c * y_c).sum() / denom)
