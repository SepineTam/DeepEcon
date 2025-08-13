#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : base.py

from abc import ABC, abstractmethod


class ErrorBase(ABC):
    error_name: str = "error_name"  # must be overwritten
    error_doc_base: str = "https://github.com/SepineTam/DeepEcon/blob/master/source/docs/deepecon/errors"
    end_with: str = ".md"

    @abstractmethod
    def __init__(self, *args, **kwargs): ...

    @abstractmethod
    def error_msg(self, *args, **kwargs) -> str: ...

    @abstractmethod
    def relative_doc_path(self) -> str: ...

    def open_error_docs(self, is_open: bool = False) -> str:
        doc_url: str = self.error_doc_base + self.relative_doc_path() + self.end_with
        if is_open:
            import webbrowser
            webbrowser.open(doc_url)
        return doc_url
