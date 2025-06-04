#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam
# @Email  : sepinetam@gmail.com
# @File   : dml.py

from abc import ABC, abstractmethod
from typing import Optional, Union

import torch


class DML(ABC):
    """
    Abstract base class for Double Machine Learning (DML) estimators.

    This class defines the common interface and high‐level workflow for
    cross‐fitting, residual computation, and causal effect estimation.

    Attributes:
        model_y_builder (callable): Factory function that accepts input_dim (int)
            and returns a torch.nn.Module for predicting Y|X.
        model_t_builder (callable): Factory function that accepts input_dim (int)
            and returns a torch.nn.Module for predicting T|X.
        discrete_treatment (bool): Whether treatment variable T is binary.
        n_splits (int): Number of folds for cross‐fitting.
        epochs (int): Number of epochs to train each nuisance model per fold.
        batch_size (int): Batch size used during training.
        lr (float): Learning rate for optimizer.
        device (str): 'cuda' or 'cpu', device on which to perform computations.

        _resid_y (torch.Tensor): Residuals of outcome Y after cross‐fitting.
        _resid_t (torch.Tensor): Residuals of treatment T after cross‐fitting.
        _theta (float): Estimated average treatment effect (ATE).
        _X (torch.Tensor): Covariates used during fitting.
    """

    def __init__(
        self,
        model_y_builder,
        model_t_builder,
        discrete_treatment: bool = True,
        n_splits: int = 5,
        epochs: int = 10,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        """
        Initialize base DML estimator with hyperparameters and builders.

        Args:
            model_y_builder (callable): Function (input_dim: int) -> torch.nn.Module
                that constructs outcome model.
            model_t_builder (callable): Function (input_dim: int) -> torch.nn.Module
                that constructs treatment model.
            discrete_treatment (bool): Whether treatment is binary.
            n_splits (int): Number of folds for cross‐fitting (K).
            epochs (int): Number of training epochs per fold for each nuisance model.
            batch_size (int): Size of minibatches for training nuisance models.
            lr (float): Learning rate for nuisance model optimizers.
            device (str): Device identifier for torch tensors ('cuda' or 'cpu').

        Raises:
            ValueError: If n_splits < 2 or lr <= 0.
            TypeError: If model builders are not callable.
        """
        pass  # Implementation will validate and store arguments

    @abstractmethod
    def fit(
        self,
        X: torch.Tensor,
        T: torch.Tensor,
        Y: torch.Tensor,
        W: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Fit the DML estimator using cross‐fitting and two‐stage residual regression.

        This method must:
          1. Split the data into K folds.
          2. For each fold:
               a. Train model_y on training partitions to predict Y|X.
               b. Train model_t on training partitions to predict T|X.
               c. Predict on held‐out fold to compute residuals for Y and T.
          3. Aggregate residuals across all folds.
          4. Perform second‐stage regression on residuals to estimate θ (ATE).

        Args:
            X (torch.Tensor): Covariate matrix of shape (n_samples, n_features).
            T (torch.Tensor): Treatment vector of shape (n_samples, 1) or (n_samples,).
            Y (torch.Tensor): Outcome vector of shape (n_samples, 1) or (n_samples,).
            W (torch.Tensor, optional): Optional additional controls of
                shape (n_samples, n_controls). Defaults to None.

        Raises:
            ValueError: If shapes of X, T, Y (and W if provided) are inconsistent.
            RuntimeError: If training fails due to device or optimizer issues.
        """
        pass

    @abstractmethod
    def ate(self, X_test: Optional[torch.Tensor] = None) -> float:
        """
        Estimate the Average Treatment Effect (ATE).

        If X_test is provided, returns the average of conditional treatment effects
        over the rows of X_test; otherwise, returns the overall ATE computed during fit.

        Args:
            X_test (torch.Tensor, optional): Covariate matrix of shape
                (m_samples, n_features) for which to compute conditional ATEs.
                If None, returns the overall ATE. Defaults to None.

        Returns:
            float: Estimated ATE (or mean of CATEs over X_test rows).

        Raises:
            RuntimeError: If fit() has not been called before calling ate().
        """
        pass

    @abstractmethod
    def effect(self, X_test: torch.Tensor) -> torch.Tensor:
        """
        Estimate Conditional Average Treatment Effects (CATEs) at given covariates.

        Args:
            X_test (torch.Tensor): Covariate matrix of shape (m_samples, n_features)
                at which to estimate CATE.

        Returns:
            torch.Tensor: A vector of length m_samples containing
                estimated treatment effects for each row of X_test.

        Raises:
            ValueError: If X_test dimensionality does not match training X.
            RuntimeError: If residuals are not available.
        """
        pass

    @abstractmethod
    def get_residuals(self) -> Union[torch.Tensor, torch.Tensor]:
        """
        Retrieve residuals for outcome and treatment computed during cross‐fitting.

        Returns:
            tuple: (resid_y, resid_t)
                resid_y (torch.Tensor): Residuals of Y of shape (n_samples,).
                resid_t (torch.Tensor): Residuals of T of shape (n_samples,).

        Raises:
            RuntimeError: If fit() has not been called yet.
        """
        pass
