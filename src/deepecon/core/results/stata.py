#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : stata.py

from .base import ResultStrMthdBase


class StataResultMthd(ResultStrMthdBase):
    name = "Stata"

    def render(self, res: "ResultBase", *args, **kwargs) -> str:
        pass
