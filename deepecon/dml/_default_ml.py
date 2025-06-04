#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam
# @Email  : sepinetam@gmail.com
# @File   : _default_ml.py

import torch
import torch.nn as nn
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from typing import Callable, Optional, Union


def default_model_y(
    input_dim: int, hidden_dims: Optional[list] = None
) -> nn.Module:
    """
    Build a default outcome model (simple MLP) for DML.

    Args:
        input_dim (int): Number of features in X.
        hidden_dims (list, optional): List of hidden layer sizes.
            If None, defaults to [64, 32].

    Returns:
        nn.Module: A PyTorch sequential MLP mapping input_dim -> [hidden_dims] -> 1.

    Example:
        >>> model = default_model_y(10, hidden_dims=[32, 16])
    """
    hidden_dims = hidden_dims or [64, 32]
    layers = []
    last_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last_dim, h))
        layers.append(nn.ReLU())
        last_dim = h
    layers.append(nn.Linear(last_dim, 1))
    return nn.Sequential(*layers)


def default_model_t(
    input_dim: int,
    hidden_dims: Optional[list] = None,
    discrete: bool = True,
) -> nn.Module:
    """
    Build a default treatment model (MLP) for DML.

    Args:
        input_dim (int): Number of features in X.
        hidden_dims (list, optional): List of hidden layer sizes.
            If None, defaults to [64, 32].
        discrete (bool): If True, final output is a logit for binary classification.
            If False, final output is a single continuous value for regression.

    Returns:
        nn.Module: A PyTorch sequential MLP mapping input_dim -> [hidden_dims] -> 1.

    Example:
        >>> model = default_model_t(10, hidden_dims=[32, 16], discrete=True)
    """
    hidden_dims = hidden_dims or [64, 32]
    layers = []
    last_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last_dim, h))
        layers.append(nn.ReLU())
        last_dim = h
    layers.append(nn.Linear(last_dim, 1))
    return nn.Sequential(*layers)


def default_sklearn_lasso(alpha: float = 1.0) -> Callable:
    """
    Return a callable that constructs a sklearn LassoCV model with given alpha.

    Args:
        alpha (float): Regularization strength for Lasso.

    Returns:
        Callable: A function that, when called with no arguments, returns a LassoCV instance.

    Example:
        >>> lasso_factory = default_sklearn_lasso(alpha=0.5)
        >>> lasso_model = lasso_factory()
    """

    def _build():
        return LassoCV(alphas=[alpha], cv=5)

    return _build


def default_sklearn_ridge(alphas: Optional[list] = None) -> Callable:
    """
    Return a callable that constructs a sklearn RidgeCV model.

    Args:
        alphas (list, optional): List of candidate alphas for RidgeCV.
            If None, defaults to [0.1, 1.0, 10.0].

    Returns:
        Callable: Function that returns a RidgeCV instance when called.

    Example:
        >>> ridge_factory = default_sklearn_ridge(alphas=[0.1, 1.0, 10.0])
        >>> ridge_model = ridge_factory()
    """
    alphas = alphas or [0.1, 1.0, 10.0]

    def _build():
        return RidgeCV(alphas=alphas, cv=5)

    return _build


def default_sklearn_forest(
    n_estimators: int = 100, max_depth: Optional[int] = None, discrete: bool = False
) -> Callable:
    """
    Return a callable that constructs a sklearn RandomForest for regression or classification.

    Args:
        n_estimators (int): Number of trees in the forest.
        max_depth (int, optional): Maximum depth of each tree. Defaults to None.
        discrete (bool): If False, returns RandomForestRegressor; if True, returns
            RandomForestClassifier.

    Returns:
        Callable: Function that returns a RandomForest model when called.

    Example:
        >>> rf_factory = default_sklearn_forest(n_estimators=50, max_depth=5, discrete=False)
        >>> rf_model = rf_factory()
    """
    def _build():
        if discrete:
            return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)

    return _build
