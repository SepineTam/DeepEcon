#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : winsor2.py

from typing import List, Optional, Dict, Tuple

import pandas as pd

from ..core.base import TransformBase
from ..core.condition import Condition


class Winsor2(TransformBase):
    name = "winsor2"

    def options(self) -> Dict[str, str]:
        return self.std_ops(
            ["X_cols", "replace"],
            add_ops={"suffix": "The suffix of the new columns, default is `_w`"}
        )

    def transform(self,
                  y_col: Optional[str] = None,
                  X_cols: Optional[List[str]] = None,
                  _if_exp: Optional[Condition] = None,
                  replace: bool = False,
                  p: Tuple[float, float] = (0.01, 0.99),
                  suffix: str = "_w",
                  *args, **kwargs) -> pd.DataFrame:
        low = self.df[X_cols].quantile(min(p))
        high = self.df[X_cols].quantile(max(p))
        if replace:
            cols = X_cols
        else:
            cols = [col+suffix for col in X_cols]
        self.df[cols] = self.df[X_cols].clip(lower=low, upper=high, axis=1)
        return self.df
