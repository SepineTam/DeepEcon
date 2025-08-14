#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : __init__.py

from .exist import FileExistError, VarExistError
from .length import LengthNotMatchError
from .not_found import (ConditionNotFoundError, FileNotFoundError,
                        OperatorNotFoundError, VarNotFoundError)

__all__ = [
    # ExistError
    "FileExistError",
    "VarExistError",

    # LengthError
    'LengthNotMatchError',

    # NotFoundError
    'ConditionNotFoundError',
    'FileNotFoundError',
    'OperatorNotFoundError',
    'VarNotFoundError',
]
