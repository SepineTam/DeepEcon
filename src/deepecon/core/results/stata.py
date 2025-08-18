#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : stata.py

from .base import ResultStrMthdBase


class StataResultMthd(ResultStrMthdBase):
    name = "stata"

    short = "-------------"
    long = "----------------------------------"
    longer = "----------------------------------------------------------------"
    black3 = "   "

    def render(self, res: "ResultBase", *args, **kwargs) -> str:
        anova = res.ANOVA
        meta = res.meta
        reg_data = res.data

        repr_str = "      Source |       SS           df       MS   "
        repr_str += self.black3
        repr_str += "Number of obs   ="
        repr_str += f"{meta.get('n'):>10}"
        repr_str += "\n"

        repr_str += self.short + "+" + self.long
        repr_str += self.black3
        f1 = meta.get("F1")
        f2 = meta.get("F2")
        f_f1_f2 = f"F({f1}, {f2})"
        repr_str += f"{f_f1_f2:<16}="
        f_2f_str = f"{meta.get('F-value'):.2f}"
        repr_str += f"{f_2f_str:>10}"
        repr_str += "\n"

        repr_str += "       Model |"
        ssm = anova.loc["Model", "SS"]
        ssm_str = str(ssm)[:10]
        repr_str += f"  {ssm_str}  "
        df_model = anova.loc["Model", "df"]
        repr_str += f"  {df_model:>6}  "
        ms_model = anova.loc['Model', 'MS']
        ms_model_str = str(ms_model)[:10]
        repr_str += f"{ms_model_str:>10}"
        repr_str += self.black3
        repr_str += "Prob > F        ="
        probF_str = f"{meta.get('ProbF'):.4f}"
        repr_str += f"{probF_str:>10}"
        repr_str += "\n"

        repr_str += "    Residual |"
        ssr = anova.loc["Residual", "SS"]
        ssr_str = str(ssr)[:10]
        repr_str += f"  {ssr_str}  "
        df_model = anova.loc["Residual", "df"]
        repr_str += f"  {df_model:>6}  "
        ms_residual = anova.loc['Residual', 'MS']
        ms_residual_str = str(ms_residual)[:10]
        repr_str += f"{ms_residual_str:>10}"
        repr_str += self.black3
        repr_str += "Prob > F        ="
        r2_str = f"{meta.get('R2'):.4f}"
        repr_str += f"{r2_str:>10}"
        repr_str += "\n"

        repr_str += self.short + "+" + self.long
        repr_str += self.black3
        repr_str += "Adj R-squared   ="
        adj_r2_str = f"{meta.get('AdjR2'):.4f}"
        repr_str += f"{adj_r2_str:>10}"
        repr_str += "\n"

        repr_str += "       Total |"
        sst = anova.loc["Total", "SS"]
        sst_str = str(sst)[:10]
        repr_str += f"  {sst_str}  "
        df_total = anova.loc["Total", "df"]
        repr_str += f"  {df_total:>6}  "
        ms_total = anova.loc['Total', 'MS']
        ms_total_str = str(ms_total)[:10]
        repr_str += f"{ms_total_str:>10}"
        repr_str += self.black3
        repr_str += "Root MSE        ="
        root_mse_str = f"{meta.get('MSE'):.4f}"
        repr_str += f"{root_mse_str:>10}"
        repr_str += "\n"
        repr_str += "\n"

        repr_str += self.short + "+" + self.longer
        repr_str += "\n"

        y_name = self.__shorter_var_name(reg_data["y_name"])
        repr_str += f" {y_name:>11} | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]"
        repr_str += "\n"

        X_names = reg_data["X_names"]
        beta = reg_data["beta"]
        stderr = reg_data["stderr"]
        t_value = reg_data["t_value"]
        p_value = reg_data["p_value"]
        ci_lower = reg_data["ci_lower"]
        ci_upper = reg_data["ci_upper"]

        n = len(X_names)
        for i in range(n):
            repr_str += f" {self.__shorter_var_name(X_names[i])} |"
            repr_str += f" {self.__shorter_float(beta[i]):>9}  "
            repr_str += f" {self.__shorter_float(stderr[i]):>9}  "
            repr_str += f" {self.process_tp_value(t_value[i])}  "
            repr_str += f" {self.process_tp_value(p_value[i])}  "
            repr_str += f" {self.__shorter_float(ci_lower[i]):>9}  "
            repr_str += f" {self.__shorter_float(ci_upper[i]):>9}  "
            repr_str += "\n"
        repr_str += self.short + "+" + self.longer
        repr_str += "\n"
        return repr_str

    @staticmethod
    def process_tp_value(value: float) -> str:
        return f"{value:.3f}"

    @staticmethod
    def __shorter_var_name(var_name: str, max_length: int = 11) -> str:
        if len(var_name) <= max_length:
            var_name = f"{var_name:>11}"
            return var_name

        prefix = var_name[:5]
        suffix = var_name[-5:]
        return f"{prefix}~{suffix}"

    @staticmethod
    def __shorter_float(data: int | float, max_length: int = 8) -> str:
        sign = "-" if data < 0 else " "
        abs_data = abs(data)

        if abs_data == 0:
            return " 0"

        # 根据数值大小选择合适的格式化方式
        if abs_data >= 1e7 or abs_data < 1e-4:
            # 使用科学计数法
            formatted = f"{abs_data:.2e}"
            parts = formatted.lower().split("e")
            mantissa = float(parts[0])
            exponent = int(parts[1])
            result = f"{sign}{mantissa:.2f}e{exponent:+03d}"
        else:
            # 使用普通小数格式
            if abs_data < 1:
                # 对于小于1的数，去掉前导0，补全小数位
                # 计算需要的小数位数，确保总长度为max_length-1（去掉符号位）
                decimal_places = max_length - 1  # 减去符号位
                formatted = f"{abs_data:.{decimal_places}f}"
                
                # 去掉前导0，例如0.123 -> .123
                if formatted.startswith("0."):
                    formatted = formatted[1:]  # 去掉前导0
                elif formatted.startswith("-0."):
                    formatted = "-" + formatted[2:]  # 去掉负号后的前导0
                
                result = f"{sign}{formatted}"
            else:
                # 对于大于等于1的数，使用科学计数法
                formatted = f"{abs_data:.2e}"
                parts = formatted.lower().split("e")
                mantissa = float(parts[0])
                exponent = int(parts[1])
                result = f"{sign}{mantissa:.2f}e{exponent:+03d}"

        # 确保总长度不超过限制，必要时截断
        if len(result) > max_length:
            # 对于科学计数法，尝试更紧凑的格式
            if "e" in result:
                formatted = f"{abs_data:.2e}"
                parts = formatted.lower().split("e")
                mantissa = float(parts[0])
                exponent = int(parts[1])
                result = f"{sign}{mantissa:.2f}e{exponent:+03d}"
            else:
                # 对于小数，直接截断
                result = result[:max_length]
                if result.endswith("."):
                    result = result[:-1]

        return result
