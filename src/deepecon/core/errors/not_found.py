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
    def relative_doc_path(self) -> str: return "not_found/README"

    def __init__(self, *args, **kwargs):
        msg = self.error_msg()
        print(msg)


class ConditionNotFoundError(NotFoundError):
    error_name = "ConditionNotFound"

    def error_msg(self) -> str:
        return (f"ConditionNotFound: Not Found Condition\n"
                f"Open Document: {self.open_error_docs()}")


class FileNotFoundError(NotFoundError):
    pass


class VarNotFoundError(NotFoundError):
    pass
