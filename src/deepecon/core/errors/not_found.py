#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : not_found.py

from .base import ErrorBase


class NotFoundError(ErrorBase):
    def relative_doc_path(self) -> str: return "not_found"


class FileNotFoundError(NotFoundError):
    def error_code(self) -> int: return 1001


class VarNotFoundError(NotFoundError):
    def error_code(self) -> int: return 1002


class ConditionNotFoundError(NotFoundError):
    def error_code(self) -> int: return 1003


class OperatorNotFoundError(NotFoundError):
    def error_code(self) -> int: return 1004
