#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : ols.py

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..core.base import EstimatorBase
from ..core.condition import Condition
from ._out import EstimatorResult


class OrdinaryLeastSquares(EstimatorBase):
    name = "ols"
    result = EstimatorResult()
    result.update_meta("ModelName", name)

    def options(self) -> Dict[str, str]:
        return self.std_ops(
            keys=["y_col", "X_cols"],
            add_ops={
                "is_cons": "Whether to add a constant column to the design matrix",
            }
        )

    def estimator(self,
                  y_col: str,
                  X_cols: List[str],
                  _if_exp: Optional[Condition] = None,
                  weight: Optional[str] = None,
                  is_cons: bool = True,
                  *args, **kwargs) -> EstimatorResult:
        # make sure all the args are exist and prepare data
        target_columns: List[str] = [y_col] + X_cols
        self.pre_process(target_columns, _if_exp)
        y_data = self.df[y_col].to_numpy(dtype=float)
        X_data = self.df[X_cols].to_numpy(dtype=float)

        if is_cons:
            X_data = np.hstack([X_data, np.ones((X_data.shape[0], 1))])
            X_cols.append("_cons")
        # n is the number of samples, k is the number of independent variables
        n, k = X_data.shape
        self.result.update_meta("n", n)

        beta_hat = np.linalg.inv(X_data.T @ X_data) @ X_data.T @ y_data
        y_hat = X_data @ beta_hat
        residuals = y_data - y_hat

        ssr = np.sum(residuals ** 2)
        sigma_squared = ssr / (n - k)
        cov_matrix = sigma_squared * np.linalg.inv(X_data.T @ X_data)
        std_errors = np.sqrt(np.diag(cov_matrix))

        t_value = beta_hat / std_errors
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_value), df=n - k))
        t_critical = stats.t.ppf(0.975, df=n - k)  # 95% CI value
        ci_lower = beta_hat - t_critical * std_errors
        ci_upper = beta_hat + t_critical * std_errors

        # Update regression result to  Result
        self.result.update_data(y_name=y_col,
                                X_names=X_cols,
                                beta=beta_hat,
                                stderr=std_errors,
                                t_value=t_value,
                                p_value=p_value,
                                ci_lower=ci_lower,
                                ci_upper=ci_upper)

        # Update ANOVA to Result
        df_resid = n - k
        if is_cons:
            y_center = y_data - y_data.mean()
            sst = float(y_center @ y_center)
            df_model = k - 1
            df_total = n - 1
        else:
            sst = float(y_data @ y_data)
            df_model = k
            df_total = n
        sse = sst - ssr
        ms_model = sse / df_model if df_model > 0 else np.nan
        ms_resid = ssr / df_resid if df_resid > 0 else np.nan
        ms_total = sst / df_total if df_total > 0 else np.nan
        self._cal_anova(df_model, df_resid, df_total, sse, ssr, sst, ms_model, ms_resid, ms_total)

        # Update meta data to result
        R2 = 1 - ssr / sst if sst > 0 else np.nan
        AdjR2 = 1 - (1 - R2) * (n - 1) / (n - k - 1) if n > k else np.nan
        F = ms_model / ms_resid if df_model > 0 and df_resid > 0 else np.nan
        pF = 1 - stats.f.cdf(F, df_model, df_resid) if np.isfinite(F) else np.nan
        rootMSE = np.sqrt(ms_resid)
        self.result.update_meta("F1", df_model)
        self.result.update_meta("F2", df_resid)
        self.result.update_meta("F-value", F)
        self.result.update_meta("ProbF", pF)
        self.result.update_meta("R2", R2)
        self.result.update_meta("AdjR2", AdjR2)
        self.result.update_meta("MSE", rootMSE)
        return self.result

    def _cal_anova(self,
                   df_model: int, df_resid: int, df_total: int,
                   sse: float, ssr: float, sst: float,
                   ms_model: float, ms_resid: float, ms_total: float):
        self.result.update_anova(key="df", index="Model",
                                 value=int(df_model))
        self.result.update_anova(key="df", index="Residual",
                                 value=int(df_resid))
        self.result.update_anova(key="df", index="Total",
                                 value=int(df_total))

        self.result.update_anova(key="SS", index="Residual",
                                 value=ssr)
        self.result.update_anova(key="SS", index="Total",
                                 value=sst)
        self.result.update_anova(key="SS", index="Model",
                                 value=sse)

        self.result.update_anova(key="MS", index="Model",
                                 value=float(ms_model) if np.isfinite(ms_model) else np.nan)
        self.result.update_anova(key="MS", index="Residual",
                                 value=float(ms_resid) if np.isfinite(ms_resid) else np.nan)
        self.result.update_anova(key="MS", index="Total",
                                 value=float(ms_total) if np.isfinite(ms_total) else np.nan)
