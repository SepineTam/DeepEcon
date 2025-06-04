#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam
# @Email  : sepinetam@gmail.com
# @File   : nonparam_dml.py

import torch
from typing import Optional, Union

from .dml import DML


class NonParamDML(DML):
    """
    Concrete DML estimator using nonparametric methods (e.g., kernel regression)
    for second‐stage residual regression.

    Inherits from DML base. After residuals are computed, fits a nonparametric
    regressor (such as Nadaraya–Watson or other kernel‐based method) on (X, resid_t)
    to predict residuals resid_y, yielding CATE estimates nonparametrically.

    Attributes:
        kernel_params (dict): Hyperparameters for kernel function (e.g., bandwidth).
        theta_hat (float): Estimated ATE (e.g., average of nonparametric CATEs).
        se_hat (float): Estimated standard error for ATE (optional).
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
        kernel_params: Optional[dict] = None,
    ) -> None:
        """
        Initialize NonParamDML with builders and kernel hyperparameters.

        Args:
            model_y_builder (callable): Factory to build torch model for outcome.
            model_t_builder (callable): Factory to build torch model for treatment.
            discrete_treatment (bool): Whether T is binary.
            n_splits (int): Number of folds for cross‐fitting.
            epochs (int): Number of epochs for nuisance model training.
            batch_size (int): Batch size for nuisance model training.
            lr (float): Learning rate for nuisance model optimizers.
            device (str): Torch device for computing.
            kernel_params (dict, optional): Dictionary of kernel hyperparameters
                (e.g., {'bandwidth': 1.0, 'kernel': 'rbf'}). Defaults to None.

        Raises:
            TypeError: If model builders are not callable.
            ValueError: If kernel_params is invalid.
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
        self.kernel_params: dict = kernel_params or {}
        self.theta_hat: Optional[float] = None
        self.se_hat: Optional[float] = None

    def fit(
        self,
        X: torch.Tensor,
        T: torch.Tensor,
        Y: torch.Tensor,
        W: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Fit the NonParamDML estimator.

        1. Perform cross‐fitting to compute residuals (resid_y, resid_t).
        2. Use a kernel‐based method to nonparametrically regress resid_y on X and resid_t:
           e.g., estimate m(X) = E[resid_y | X], weight by resid_t.
        3. Compute CATE_i = m(X_i) * resid_t_i for each sample (depending on method).
        4. Set theta_hat = average of CATE_i.
        5. Optionally compute se_hat via Bootstrap.

        Args:
            X (torch.Tensor): Covariates, shape (n_samples, n_features).
            T (torch.Tensor): Treatment, shape (n_samples, 1) or (n_samples,).
            Y (torch.Tensor): Outcome, shape (n_samples, 1) or (n_samples,).
            W (torch.Tensor, optional): Additional controls, shape
                (n_samples, n_controls). Defaults to None.

        Raises:
            ValueError: If input shapes are inconsistent.
            RuntimeError: If kernel regression fails or residuals missing.
        """
        pass

    def ate(self, X_test: Optional[torch.Tensor] = None) -> float:
        """
        Return the estimated Average Treatment Effect (ATE).

        If X_test is None, returns average of CATEs computed during fit().
        If X_test is provided, uses the fitted kernel regressor to estimate
        CATEs at X_test and returns their mean.

        Args:
            X_test (torch.Tensor, optional): Covariate matrix of shape
                (m_samples, n_features). Defaults to None.

        Returns:
            float: Estimated ATE or mean of CATEs over X_test.

        Raises:
            RuntimeError: If fit() hasn't been called or kernel model not available.
        """
        pass

    def effect(self, X_test: torch.Tensor) -> torch.Tensor:
        """
        Estimate conditional treatment effects (CATEs) at given covariates X_test
        using the fitted kernel regressor.

        Args:
            X_test (torch.Tensor): Covariate matrix of shape (m_samples, n_features).

        Returns:
            torch.Tensor: Tensor of length m_samples containing CATE estimates.

        Raises:
            RuntimeError: If fit() hasn't been called or kernel model not available.
            ValueError: If X_test dimensionality does not match training X.
        """
        pass

    def get_residuals(self) -> Union[torch.Tensor, torch.Tensor]:
        """
        Retrieve residuals (resid_y, resid_t) computed during cross‐fitting.

        Returns:
            tuple: (resid_y, resid_t)
                resid_y (torch.Tensor): Residuals of Y of shape (n_samples,).
                resid_t (torch.Tensor): Residuals of T of shape (n_samples,).

        Raises:
            RuntimeError: If fit() has not been called and no residuals exist.
        """
        pass
