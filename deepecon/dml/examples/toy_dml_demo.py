#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam
# @Email  : sepinetam@gmail.com
# @File   : toy_dml_demo.py

"""
Example script demonstrating how to use DeepEcon's DML subpackage.

This toy demo generates synthetic data, fits various DML estimators,
and prints the estimated ATE. Intended for verification and testing.
"""

import torch

from .._utils import set_random_seed, get_device
from .._default_ml import default_model_y, default_model_t
from ..linear_dml import LinearDML
from ..causal_forest_dml import CausalForestDML
from ..sparse_linear_dml import SparseLinearDML

# If using NonParamDML or KernelDML, import them similarly:
# from ..nonparam_dml import NonParamDML
# from ..kernel_dml import KernelDML


def generate_synthetic_data(
    n: int = 1000, dim: int = 5, seed: int = 0
) -> torch.Tensor:
    """
    Generate synthetic data for testing DML estimators.

    The data‐generating process:
        X ~ N(0, I_dim)
        True CATE: theta(X) = 2 * X[:, 0]
        T ~ Bernoulli(sigmoid(X @ w_t + noise))
        Y = theta(X) * T + f(X) + noise

    Args:
        n (int): Number of samples to generate.
        dim (int): Number of covariate features.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            X (torch.Tensor): Covariate matrix shape (n, dim).
            T (torch.Tensor): Treatment vector shape (n, 1).
            Y (torch.Tensor): Outcome vector shape (n, 1).

    Raises:
        ValueError: If n <= 0 or dim <= 0.
    """
    if n <= 0:
        raise ValueError("Number of samples n must be positive.")
    if dim <= 0:
        raise ValueError("Dimension dim must be positive.")
    set_random_seed(seed)
    X = torch.randn(n, dim)
    # True heterogeneous treatment effect: theta_i = 2 * X[i, 0]
    true_theta = (2.0 * X[:, 0]).view(n, 1)
    # Generate treatment T using a logistic model on first two features
    logits = X[:, 1].unsqueeze(1) + 0.5 * X[:, 2].unsqueeze(1)
    prob = torch.sigmoid(logits)
    T = torch.bernoulli(prob).view(n, 1)
    # Baseline outcome f(X) = 0.1 * sum of features from index 3 onward
    f_X = 0.1 * X[:, 3:].sum(dim=1, keepdim=True)
    noise_Y = 0.1 * torch.randn(n, 1)
    Y = true_theta * T + f_X + noise_Y
    return X, T, Y


def main() -> None:
    """
    Main function to run the toy DML demonstration.

    1. Generate synthetic data.
    2. Initialize and fit LinearDML, CausalForestDML, SparseLinearDML.
    3. Print estimated ATEs for each estimator.
    """
    # Generate data
    X, T, Y = generate_synthetic_data(n=2000, dim=10, seed=42)
    device = get_device()
    X = X.to(device)
    T = T.to(device)
    Y = Y.to(device)

    # Prepare nuisance model builders
    model_y_builder = lambda d: default_model_y(d, hidden_dims=[32, 16]).to(device)
    model_t_builder = lambda d: default_model_t(d, hidden_dims=[32, 16], discrete=True).to(device)

    # 1. LinearDML
    linear_dml = LinearDML(
        model_y_builder=model_y_builder,
        model_t_builder=model_t_builder,
        discrete_treatment=True,
        n_splits=5,
        epochs=5,
        batch_size=128,
        lr=1e-3,
        device=device,
    )
    linear_dml.fit(X, T, Y)
    ate_linear = linear_dml.ate()
    print(f"LinearDML estimated ATE: {ate_linear:.4f}")

    # 2. CausalForestDML
    cf_dml = CausalForestDML(
        model_y_builder=model_y_builder,
        model_t_builder=model_t_builder,
        discrete_treatment=True,
        n_splits=5,
        epochs=5,
        batch_size=128,
        lr=1e-3,
        device=device,
        n_trees=100,
        max_depth=5,
    )
    cf_dml.fit(X, T, Y)
    ate_cf = cf_dml.ate()
    print(f"CausalForestDML estimated ATE: {ate_cf:.4f}")

    # 3. SparseLinearDML
    sparse_dml = SparseLinearDML(
        model_y_builder=model_y_builder,
        model_t_builder=model_t_builder,
        discrete_treatment=True,
        n_splits=5,
        epochs=5,
        batch_size=128,
        lr=1e-3,
        device=device,
        l1_alpha=0.1,
    )
    sparse_dml.fit(X, T, Y)
    ate_sparse = sparse_dml.ate()
    print(f"SparseLinearDML estimated ATE: {ate_sparse:.4f}")

    # Additional estimators (NonParamDML, KernelDML) can be tested similarly.


if __name__ == "__main__":
    main()
