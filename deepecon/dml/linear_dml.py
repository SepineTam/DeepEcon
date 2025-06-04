#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam
# @Email  : sepinetam@gmail.com
# @File   : linear_dml.py

import torch
from typing import Optional, Union

from .dml import DML


class LinearDML(DML):
    """
    Concrete DML estimator using linear second‐stage regression.

    Inherits from the abstract DML base class and implements:
      1. Cross‐fitting of nuisance models.
      2. Computation of residuals for Y and T.
      3. Second‐stage linear regression to estimate treatment effect θ.
      4. Optionally estimate CATEs via linear model on residuals.

    Attributes:
        theta_hat (float): Point estimate of Average Treatment Effect after fit.
        se_hat (float): Estimated standard error for ATE (optional).
        coef_cate (torch.Tensor): Coefficients for linear CATE model if requested.
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
        Initialize LinearDML with nuisance model builders and hyperparameters.

        Args:
            model_y_builder (callable): Callable(input_dim: int) -> torch.nn.Module
                for outcome model.
            model_t_builder (callable): Callable(input_dim: int) -> torch.nn.Module
                for treatment model.
            discrete_treatment (bool): Whether treatment variable T is binary.
            n_splits (int): Number of folds for cross‐fitting (K).
            epochs (int): Number of epochs for training each nuisance model per fold.
            batch_size (int): Batch size for training nuisance models.
            lr (float): Learning rate for nuisance model optimizers.
            device (str): Device string for torch computations ('cuda' or 'cpu').

        Raises:
            TypeError: If model builders are not callable.
            ValueError: If n_splits < 2 or lr <= 0.
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
        # After fit() is called:
        # self.theta_hat: float
        # self.se_hat: Optional[float]
        # self.coef_cate: Optional[torch.Tensor]
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
        Fit the LinearDML estimator.

        1. Perform K‐fold cross‐fitting for nuisance models (model_y, model_t).
        2. Compute residuals resid_y = Y - E_hat[Y|X] and resid_t = T - E_hat[T|X].
        3. Run second‐stage OLS regression of resid_y on resid_t (and optional W)
           to estimate theta_hat (ATE) and standard error se_hat.
        4. If W is provided (additional controls), include W in second‐stage regression.

        Args:
            X (torch.Tensor): Covariate matrix, shape (n_samples, n_features).
            T (torch.Tensor): Treatment array, shape (n_samples, 1) or (n_samples,).
            Y (torch.Tensor): Outcome array, shape (n_samples, 1) or (n_samples,).
            W (torch.Tensor, optional): Additional controls of shape
                (n_samples, n_controls). Defaults to None.

        Raises:
            ValueError: If input shapes are incompatible.
            RuntimeError: If cross‐fitting or second‐stage regression fails.
        """
        pass

    def ate(self, X_test: Optional[torch.Tensor] = None) -> float:
        """
        Return the Average Treatment Effect (ATE) or average of CATEs over X_test.

        If X_test is None, returns the point estimate theta_hat computed during fit().
        If X_test is provided, uses the fitted CATE linear model (coef_cate)
        to compute per‐row effects, then returns their mean.

        Args:
            X_test (torch.Tensor, optional): Covariate matrix for which to compute
                conditional ATEs, shape (m_samples, n_features). Defaults to None.

        Returns:
            float: Estimated ATE (or mean of CATEs over X_test if provided).

        Raises:
            RuntimeError: If fit() has not been called or coef_cate is None when X_test is provided.
        """
        pass

    def effect(self, X_test: torch.Tensor) -> torch.Tensor:
        """
        Estimate conditional treatment effects (CATEs) at input covariates X_test.

        Uses the second‐stage linear model coefficients (coef_cate) learned in fit()
        to compute CATE_i = X_test[i] @ coef_cate (plus intercept if applicable).

        Args:
            X_test (torch.Tensor): Covariate matrix of shape (m_samples, n_features)
                at which to estimate treatment effects.

        Returns:
            torch.Tensor: Tensor of length m_samples containing estimated CATEs.

        Raises:
            RuntimeError: If fit() has not been called or coef_cate is None.
            ValueError: If X_test dimensionality does not match training X.
        """
        pass

    def get_residuals(self) -> Union[torch.Tensor, torch.Tensor]:
        """
        Retrieve residuals for outcome and treatment from cross‐fitting.

        Returns:
            tuple: (resid_y, resid_t)
                resid_y (torch.Tensor): Residuals of Y, shape (n_samples,).
                resid_t (torch.Tensor): Residuals of T, shape (n_samples,).

        Raises:
            RuntimeError: If fit() has not been called yet and no residuals exist.
        """
        pass
