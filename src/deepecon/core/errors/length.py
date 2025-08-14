#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : length.py

from .base import ErrorBase


class LengthError(ErrorBase):
    def relative_doc_path(self) -> str: return "length"


class LengthNotMatchError(LengthError):
    def error_code(self) -> int: return 2001
