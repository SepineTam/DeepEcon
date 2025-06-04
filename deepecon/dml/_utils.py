#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam
# @Email  : sepinetam@gmail.com
# @File   : _utils.py

import random
import numpy as np
import torch
from sklearn.model_selection import KFold
from typing import List, Tuple, Optional


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for Python random, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to set for all random generators.

    Raises:
        ValueError: If seed is negative.
    """
    if seed < 0:
        raise ValueError("Seed must be non‐negative.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """
    Return the default device string: 'cuda' if available, else 'cpu'.

    Returns:
        str: 'cuda' or 'cpu'.

    Example:
        >>> device = get_device()
        >>> model.to(device)
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def split_indices(
    n_samples: int, n_splits: int, shuffle: bool = True, seed: Optional[int] = None
) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/validation indices for K‐fold cross‐fitting.

    Args:
        n_samples (int): Total number of samples.
        n_splits (int): Number of folds (K) for cross‐splitting.
        shuffle (bool): Whether to shuffle indices before splitting. Defaults to True.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        List[Tuple[List[int], List[int]]]: A list of length n_splits containing
            (train_idx, val_idx) pairs, where each is a list of sample indices.

    Raises:
        ValueError: If n_splits < 2 or n_splits > n_samples.
    """
    if not (2 <= n_splits <= n_samples):
        raise ValueError("n_splits must be >=2 and <= n_samples.")
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    splits = []
    for train_idx, val_idx in kf.split(range(n_samples)):
        splits.append((train_idx.tolist(), val_idx.tolist()))
    return splits


def compute_theta_from_residuals(
    resid_t: torch.Tensor, resid_y: torch.Tensor
) -> float:
    """
    Compute ATE estimate θ from residuals via closed‐form formula:
        θ = sum(resid_t * resid_y) / sum(resid_t^2)

    Args:
        resid_t (torch.Tensor): Residuals of T, shape (n,).
        resid_y (torch.Tensor): Residuals of Y, shape (n,).

    Returns:
        float: Estimated treatment effect θ.

    Raises:
        ValueError: If resid_t and resid_y have different shapes or zero variance in resid_t.
    """
    if resid_t.shape != resid_y.shape:
        raise ValueError("resid_t and resid_y must have the same shape.")
    denom = (resid_t * resid_t).sum().item()
    if denom == 0:
        raise ValueError("Zero variance in resid_t; cannot compute θ.")
    numer = (resid_t * resid_y).sum().item()
    return numer / denom


def bootstrap_ate_ci(
    resid_t: torch.Tensor,
    resid_y: torch.Tensor,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Compute Bootstrap confidence interval for ATE via resampling residuals.

    Args:
        resid_t (torch.Tensor): Residuals of T, shape (n,).
        resid_y (torch.Tensor): Residuals of Y, shape (n,).
        n_bootstrap (int): Number of bootstrap samples. Defaults to 1000.
        alpha (float): Significance level for two‐sided interval (e.g., 0.05 for 95% CI).
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        Tuple[float, float]: (lower_bound, upper_bound) of the confidence interval.

    Raises:
        ValueError: If n_bootstrap <= 0 or alpha not in (0,1).
    """
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be > 0.")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1.")
    n = resid_t.shape[0]
    # Convert to NumPy for random sampling
    resid_t_np = resid_t.detach().cpu().numpy().reshape(-1)
    resid_y_np = resid_y.detach().cpu().numpy().reshape(-1)
    rng = np.random.RandomState(seed)
    thetas = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        t_sample = torch.from_numpy(resid_t_np[idx])
        y_sample = torch.from_numpy(resid_y_np[idx])
        theta_b = compute_theta_from_residuals(t_sample, y_sample)
        thetas.append(theta_b)
    thetas = np.array(thetas)
    lower = float(np.percentile(thetas, 100 * (alpha / 2)))
    upper = float(np.percentile(thetas, 100 * (1 - alpha / 2)))
    return lower, upper
