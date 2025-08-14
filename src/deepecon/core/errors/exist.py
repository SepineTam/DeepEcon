#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : exist.py

from .base import ErrorBase


class ExistError(ErrorBase):
    def relative_doc_path(self) -> str: return "exist"


class FileExistError(ExistError):
    def error_code(self) -> int: return 3001


class VarExistError(ExistError):
    def error_code(self) -> int: return 3002
