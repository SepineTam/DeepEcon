#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : condition.py

from __future__ import annotations

import operator as _op
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Optional, Sequence, Union

import pandas as pd


class CondBase(ABC):
    @abstractmethod
    def to_mask(self, df: pd.DataFrame) -> pd.Series: ...

    def __and__(self, other: "CondBase") -> "CondBase":
        return And(self, other)

    def __or__(self, other: "CondBase") -> "CondBase":
        return Or(self, other)

    def __invert__(self) -> "CondBase":
        return Not(self)


@dataclass(frozen=True)
class And(CondBase):
    left: CondBase
    right: CondBase

    def to_mask(self, df: pd.DataFrame) -> pd.Series:
        return self.left.to_mask(df) & self.right.to_mask(df)


@dataclass(frozen=True)
class Or(CondBase):
    left: CondBase
    right: CondBase

    def to_mask(self, df: pd.DataFrame) -> pd.Series:
        return self.left.to_mask(df) | self.right.to_mask(df)


@dataclass(frozen=True)
class Not(CondBase):
    inner: CondBase

    def to_mask(self, df: pd.DataFrame) -> pd.Series:
        return ~self.inner.to_mask(df)


Op = Literal["==", "!=", ">", ">=", "<", "<="]
_OP_MAP = {
    "==": _op.eq, "!=": _op.ne,
    ">": _op.gt, ">=": _op.ge,
    "<": _op.lt, "<=": _op.le,
}


@dataclass(frozen=True)
class Col:
    name: str

    def _cmp(self, op: Op, other: Any) -> "Compare":
        return Compare(self, op, other)

    def __eq__(self, other: Any) -> "Compare":  # type: ignore[override]
        return self._cmp("==", other)

    def __ne__(self, other: Any) -> "Compare":  # type: ignore[override]
        return self._cmp("!=", other)

    def __gt__(self, other: Any) -> "Compare":
        return self._cmp(">", other)

    def __ge__(self, other: Any) -> "Compare":
        return self._cmp(">=", other)

    def __lt__(self, other: Any) -> "Compare":
        return self._cmp("<", other)

    def __le__(self, other: Any) -> "Compare":
        return self._cmp("<=", other)

    def isin(self, values: Sequence[Any]) -> "InSet":
        return InSet(self, tuple(values))

    def notin(self, values: Sequence[Any]) -> "NotInSet":
        return NotInSet(self, tuple(values))


@dataclass(frozen=True)
class Compare(CondBase):
    col: Col
    op: Op
    value: Any

    def to_mask(self, df: pd.DataFrame) -> pd.Series:
        fn = _OP_MAP[self.op]
        return fn(df[self.col.name], self.value)


@dataclass(frozen=True)
class InSet(CondBase):
    col: Col
    values: Sequence[Any]

    def to_mask(self, df: pd.DataFrame) -> pd.Series:
        return df[self.col.name].isin(self.values)


@dataclass(frozen=True)
class NotInSet(CondBase):
    col: Col
    values: Sequence[Any]

    def to_mask(self, df: pd.DataFrame) -> pd.Series:
        return ~df[self.col.name].isin(self.values)


@dataclass(frozen=True)
class MaskCond(CondBase):
    mask: pd.Series

    def to_mask(self, df: pd.DataFrame) -> pd.Series:
        return self.mask.reindex(df.index).astype(bool)


@dataclass(frozen=True)
class CallableCond(CondBase):
    fn: Callable[[pd.DataFrame], pd.Series]

    def to_mask(self, df: pd.DataFrame) -> pd.Series:
        return self.fn(df).astype(bool)


@dataclass(frozen=True)
class QueryCond(CondBase):
    expr: str
    engine: Literal["python", "numexpr"] = "python"
    local_vars: Mapping[str, Any] = field(default_factory=dict)

    def to_mask(self, df: pd.DataFrame) -> pd.Series:
        idx = df.query(
            self.expr,
            engine=self.engine,
            local_dict=dict(
                self.local_vars)).index
        return df.index.isin(idx)


def col(name: str) -> Col:
    return Col(name)


def as_cond(c: Union[CondBase, str, pd.Series, Callable[[
        pd.DataFrame], pd.Series], tuple[str, Op, Any]]) -> CondBase:
    if isinstance(c, CondBase):
        return c
    if isinstance(c, str):
        return QueryCond(c)
    if isinstance(c, pd.Series):
        return MaskCond(c)
    if callable(c):
        return CallableCond(c)
    if isinstance(c, tuple) and len(c) == 3:
        k, op, v = c
        return Compare(Col(k), op, v)
    raise TypeError("Unsupported condition input")


ConditionLike = Union[
    "Condition",
    "CondBase",
    str,
    pd.Series,
    Callable[[pd.DataFrame], pd.Series],
    tuple[str, Literal["==", "!=", ">", ">=", "<", "<="], Any],
]


@dataclass(frozen=True)
class Condition:
    _inner: "CondBase"
    _na_as_true: bool = False

    def __init__(
        self,
        expr: ConditionLike,
        *,
        engine: Literal["python", "numexpr"] = "python",
        local_vars: Optional[Mapping[str, Any]] = None,
        na_as_true: bool = False,
    ):
        object.__setattr__(self, "_na_as_true", na_as_true)

        if isinstance(expr, Condition):
            object.__setattr__(self, "_inner", expr._inner)
        elif isinstance(expr, str):
            object.__setattr__(
                self,
                "_inner",
                QueryCond(
                    expr,
                    engine=engine,
                    local_vars=local_vars or {}))
        elif isinstance(expr, pd.Series):
            object.__setattr__(self, "_inner", MaskCond(expr))
        elif callable(expr):
            object.__setattr__(self, "_inner", CallableCond(expr))
        elif isinstance(expr, tuple) and len(expr) == 3:
            k, op, v = expr
            object.__setattr__(self, "_inner", Compare(Col(k), op, v))
        elif isinstance(expr, CondBase):
            object.__setattr__(self, "_inner", expr)
        else:
            raise TypeError(f"Unsupported condition input: {type(expr)}")

    def to_mask(self, df: pd.DataFrame) -> pd.Series:
        return self._inner.to_mask(df).fillna(self._na_as_true).astype(bool)

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return self.to_mask(df)

    def __and__(self, other: ConditionLike) -> "Condition":
        o = other if isinstance(other, Condition) else Condition(other)
        # type: ignore[name-defined]
        from_existing = And(self._inner, o._inner)
        return Condition(from_existing, na_as_true=self._na_as_true)

    def __or__(self, other: ConditionLike) -> "Condition":
        o = other if isinstance(other, Condition) else Condition(other)
        # type: ignore[name-defined]
        from_existing = Or(self._inner, o._inner)
        return Condition(from_existing, na_as_true=self._na_as_true)

    def __invert__(self) -> "Condition":
        # type: ignore[name-defined]
        from_existing = Not(self._inner)
        return Condition(from_existing, na_as_true=self._na_as_true)

    def __repr__(self) -> str:
        return f"Condition({self._inner!r}, na_as_true={self._na_as_true})"
