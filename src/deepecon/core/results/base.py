#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : base.py

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..base import ResultBase


class ResultStrMthdBase(ABC):
    name: str

    @abstractmethod
    def render(self, res: "ResultBase", *args, **kwargs) -> str: ...
