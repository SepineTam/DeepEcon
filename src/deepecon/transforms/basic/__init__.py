#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : __init__.py

import math
import re
from typing import List

import numpy as np
import pandas as pd

from ...core.errors.not_found import VarNotFoundError
from .base import BasicMathBase


class BasicMath(BasicMathBase):
    name = "basic_math"

    def _evaluate_expression(self, expr: str) -> pd.Series:
        """
        Parse and evaluate mathematical expression

        Args:
            expr (str): Mathematical expression string

        Returns:
            Calculation result as Series
        """
        # Replace operators to Python-supported format
        expr = expr.replace('^', '**')

        # Find all variable names (column names)
        variables = self._extract_variables(expr)

        # Create variable mapping
        var_dict = {}
        for var in variables:
            if var in self.df.columns:
                var_dict[var] = self.df[var]
            else:
                # If variable does not exist, try to convert to number
                try:
                    var_dict[var] = float(var)
                except ValueError:
                    raise VarNotFoundError(
                        f"Variable '{var}' not found in DataFrame")

        # Define safe mathematical functions
        safe_dict = {
            'log': np.log,
            'exp': np.exp,
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'sqrt': np.sqrt,
            'abs': np.abs,
            'max': np.maximum,
            'min': np.minimum,
            'pi': math.pi,
            'e': math.e
        }

        # Merge variables and functions
        eval_dict = {**var_dict, **safe_dict}

        try:
            # Calculate expression
            result = eval(expr, {"__builtins__": {}}, eval_dict)

            # Convert scalar to Series if needed
            if np.isscalar(result):
                result = pd.Series([result] * len(self.df))

            return result

        except ZeroDivisionError:
            # Handle division by zero error
            return pd.Series([np.nan] * len(self.df))
        except Exception as e:
            raise ValueError(f"Expression calculation error: {str(e)}")

    def _extract_variables(self, expr: str) -> List[str]:
        """
        Extract variable names from expression

        Args:
            expr (str): Mathematical expression

        Returns:
            List of variable names
        """
        # Remove non-variable characters
        expr_clean = re.sub(
            r'[\s\+\-\*\/\^\(\)\,\=\<\>\!\&\|\%\~\`\@\#\$\&\*\(\)\[\]\{\}\;\:\'\"\?\<\>\,\.\|\\\/]+',
            ' ',
            expr)

        # Split and filter variable names
        tokens = expr_clean.split()
        variables = []

        for token in tokens:
            # Check if it's a numeric constant
            if not token.replace(
                    '.',
                    '',
                    1).isdigit() and token not in [
                'pi',
                'e'
            ]:
                # Check if it's a mathematical function
                if token not in [
                    'log',
                    'exp',
                    'sin',
                    'cos',
                    'tan',
                    'sqrt',
                    'abs',
                    'max',
                    'min'
                ]:
                    variables.append(token)

        return list(set(variables))  # Remove duplicates


__all__ = ["BasicMath"]
