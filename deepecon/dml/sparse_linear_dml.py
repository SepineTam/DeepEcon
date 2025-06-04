#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam
# @Email  : sepinetam@gmail.com
# @File   : sparse_linear_dml.py

import torch
from typing import Optional, Union

from .dml import DML


class SparseLinearDML(DML):
    """
    Concrete DML estimator that uses L1‐regularized (sparse) linear regression
    in the second‐stage to estimate causal effects in high‐dimensional settings.

    Inherits from the abstract DML base class. After computing residuals via
    cross‐fitting, fits a Lasso (L1) regression of residual_y on residual_t and/or W
    to select sparse coefficients for CATE.

    Attributes:
        theta_hat (float): Point estimate of ATE after fit.
        se_hat (float): Estimated standard error for ATE (optional).
        coef_cate (torch.Tensor): Sparse coefficient vector for CATE model if available.
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
        l1_alpha: float = 1.0,
    ) -> None:
        """
        Initialize SparseLinearDML with builders, hyperparameters, and L1 penalty.

        Args:
            model_y_builder (callable): Callable(input_dim: int) -> torch.nn.Module
                for outcome model.
            model_t_builder (callable): Callable(input_dim: int) -> torch.nn.Module
                for treatment model.
            discrete_treatment (bool): Whether treatment variable T is binary.
            n_splits (int): Number of folds for cross‐fitting.
            epochs (int): Number of epochs to train each nuisance model per fold.
            batch_size (int): Batch size used during training nuisance models.
            lr (float): Learning rate for nuisance model optimizers.
            device (str): Torch device for computations ('cuda' or 'cpu').
            l1_alpha (float): Regularization strength (L1 penalty) for second‐stage regression.

        Raises:
            ValueError: If l1_alpha < 0 or n_splits < 2.
            TypeError: If model builders are not callable.
        """
        super().__init__(
            model_y_builder=model_y_builder,
            model_t_builder=model_t_builder,
            discrete_treatment=discrete_treatment,
            n_splits=n_splits,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
        )
        self.l1_alpha: float = l1_alpha
        self.theta_hat: Optional[float] = None
        self.se_hat: Optional[float] = None
        self.coef_cate: Optional[torch.Tensor] = None

    def fit(
        self,
        X: torch.Tensor,
        T: torch.Tensor,
        Y: torch.Tensor,
        W: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Fit the SparseLinearDML estimator.

        1. Perform cross‐fitting on nuisance models to obtain residuals for Y and T.
        2. Stack resid_t (and W if provided) as features in second‐stage.
        3. Fit a Lasso (L1) regression to predict resid_y from [resid_t, W].
        4. Extract theta_hat (coefficient on resid_t) and coef_cate (sparse vector
           including possible coefficients for W interactions if implemented).
        5. Optionally compute standard errors via debiased Lasso procedures or Bootstrap.

        Args:
            X (torch.Tensor): Covariates tensor, shape (n_samples, n_features).
            T (torch.Tensor): Treatment tensor, shape (n_samples, 1) or (n_samples,).
            Y (torch.Tensor): Outcome tensor, shape (n_samples, 1) or (n_samples,).
            W (torch.Tensor, optional): Optional additional controls, shape
                (n_samples, n_controls). Defaults to None.

        Raises:
            ValueError: If input dimensions mismatch.
            RuntimeError: If Lasso regression fails to converge or residuals are missing.
        """
        pass

    def ate(self, X_test: Optional[torch.Tensor] = None) -> float:
        """
        Return the estimated Average Treatment Effect (ATE) or average of CATEs
        over X_test using the fitted Lasso second‐stage model.

        If X_test is None, returns theta_hat (coefficient on resid_t) computed
        during fit(). If X_test is provided, uses coef_cate to compute
        conditional treatment effects and returns their mean.

        Args:
            X_test (torch.Tensor, optional): Covariate matrix of shape
                (m_samples, n_features). Defaults to None.

        Returns:
            float: Estimated ATE or mean of CATEs over X_test.

        Raises:
            RuntimeError: If fit() not called or coef_cate is None when X_test provided.
        """
        pass

    def effect(self, X_test: torch.Tensor) -> torch.Tensor:
        """
        Estimate conditional treatment effects (CATEs) at given covariates X_test.

        Uses the sparse coefficient vector coef_cate computed during fit() to
        predict CATE_i = X_test[i] @ coef_cate (accounting for intercept if included).

        Args:
            X_test (torch.Tensor): Covariate matrix of shape (m_samples, n_features).

        Returns:
            torch.Tensor: Tensor of shape (m_samples,) containing CATEs.

        Raises:
            RuntimeError: If fit() not called or coef_cate is None.
            ValueError: If X_test dimensionality mismatches training X.
        """
        pass

    def get_residuals(self) -> Union[torch.Tensor, torch.Tensor]:
        """
        Retrieve residuals computed for Y and T during cross‐fitting.

        Returns:
            tuple: (resid_y, resid_t)
                resid_y (torch.Tensor): Residuals of Y of shape (n_samples,).
                resid_t (torch.Tensor): Residuals of T of shape (n_samples,).

        Raises:
            RuntimeError: If fit() has not been called and residuals are absent.
        """
        pass
