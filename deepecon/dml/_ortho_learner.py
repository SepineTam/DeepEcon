#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam
# @Email  : sepinetam@gmail.com
# @File   : _ortho_learner.py

import torch
from abc import ABC, abstractmethod
from typing import Optional, Union

from .dml import DML


class OrthogonalLearner(DML, ABC):
    """
    Abstract base class for generic orthogonal learners (generalization of DML).

    Extends DML to allow alternative algorithms that rely on orthogonalization
    beyond simple residual regression (e.g., two‐stage methods, cross‐fitting).

    Attributes:
        _resid_y (torch.Tensor): Outcome residuals.
        _resid_t (torch.Tensor): Treatment residuals.
        _X (torch.Tensor): Covariates matrix.
        _theta (float): Estimated target parameter (ATE or similar).
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
        Initialize OrthogonalLearner with nuisance builders and training parameters.

        Args:
            model_y_builder (callable): Callable(input_dim: int) -> torch.nn.Module.
            model_t_builder (callable): Callable(input_dim: int) -> torch.nn.Module.
            discrete_treatment (bool): Whether T is binary.
            n_splits (int): Number of folds for cross‐fitting.
            epochs (int): Epochs per fold for nuisance training.
            batch_size (int): Batch size for nuisance training.
            lr (float): Learning rate for nuisance optimizers.
            device (str): Device identifier for torch.

        Raises:
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
        self._resid_y: Optional[torch.Tensor] = None
        self._resid_t: Optional[torch.Tensor] = None
        self._theta: Optional[float] = None

    def fit(
        self,
        X: torch.Tensor,
        T: torch.Tensor,
        Y: torch.Tensor,
        W: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Fit the orthogonal learner to estimate target parameter(s).

        Must implement:
          1. Cross‐fitting stage for nuisance models (compute resid_y, resid_t).
          2. Orthogonal estimation stage specific to algorithm (e.g., partialling out
             W or handling instrumental variables).

        Args:
            X (torch.Tensor): Covariate matrix of shape (n_samples, n_features).
            T (torch.Tensor): Treatment variable tensor.
            Y (torch.Tensor): Outcome variable tensor.
            W (torch.Tensor, optional): Additional controls. Defaults to None.

        Raises:
            NotImplementedError: If subclass does not override this method.
        """
        raise NotImplementedError

    def ate(self, X_test: Optional[torch.Tensor] = None) -> float:
        """
        Return the estimated average effect (ATE or similar quantity).

        Subclasses should override if CATE is also supported.

        Args:
            X_test (torch.Tensor, optional): Covariates for conditional effect.
        """
        if self._theta is None:
            raise RuntimeError("Must call fit() before calling ate().")
        return self._theta

    def effect(self, X_test: torch.Tensor) -> torch.Tensor:
        """
        Estimate conditional effect (CATE) if supported by subclass.

        Args:
            X_test (torch.Tensor): Covariate matrix for which to compute CATEs.

        Returns:
            torch.Tensor: CATE estimates of length m_samples.

        Raises:
            NotImplementedError: If subclass does not implement conditional effect.
        """
        raise NotImplementedError

    def get_residuals(self) -> Union[torch.Tensor, torch.Tensor]:
        """
        Retrieve residuals computed by cross‐fitting.

        Returns:
            tuple: (resid_y, resid_t)

        Raises:
            RuntimeError: If residuals are not available because fit() wasn't called.
        """
        if self._resid_y is None or self._resid_t is None:
            raise RuntimeError("Residuals are not available; call fit() first.")
        return self._resid_y, self._resid_t
