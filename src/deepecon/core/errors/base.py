#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : base.py

import logging
from abc import ABC, abstractmethod


class ErrorBase(Exception, ABC):
    error_doc_base: str = "https://github.com/SepineTam/DeepEcon/tree/master/source/docs/deepecon/errors/"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logging.warning(self.error_msg())

    def error_msg(self, *args, **kwargs) -> str:
        return (f"\nError Code: {self.error_code()}"
                f"\nOpen Document: {self.open_error_docs()}")

    @abstractmethod
    def relative_doc_path(self) -> str: ...

    @abstractmethod
    def error_code(self) -> int: ...

    def open_error_docs(self, is_open: bool = False) -> str:
        doc_url: str = self.error_doc_base + self.relative_doc_path()
        if is_open:
            import webbrowser
            webbrowser.open(doc_url)
        return doc_url
