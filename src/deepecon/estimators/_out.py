#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : _out.py

from typing import List

from ..core.base import ResultBase


class EstimatorResult(ResultBase):
    def meta_keys(self) -> List[str]:
        return ["ModelName", "n", "F1", "F2", "F-value", "ProbF", "R2", "AdjR2", "MSE"]

    def data_keys(self) -> List[str]:
        return ["y", "x", "beta", "stderr", "t_value", "p_value", "ci_low", "ci_high"]
