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
    """Class for rendering results in Stata-like format."""
    
    name = "stata"

    short = "-------------"
    long = "----------------------------------"
    longer = "----------------------------------------------------------------"
    black3 = "   "

    def render(self, res: "ResultBase", *args, **kwargs) -> str:
        """Render the result in a Stata-like format.
        
        Args:
            res: The result object to render.
            
        Returns:
            A string representation of the result in Stata-like format.
        """
        anova = res.ANOVA
        meta = res.meta
        reg_data = res.data

        repr_str = f"Date: {meta.get('Date')} Time: {meta.get('time')}"
        repr_str += "\n"
        repr_str += self.short + "+" + self.longer
        repr_str += "\n"
        repr_str += "      Source |       SS           df       MS   "
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
        root_mse_str = f"{meta.get('MSE'):.4f}"[0:6]
        repr_str += f"{root_mse_str:>10}"
        repr_str += "\n"
        repr_str += "\n"

        repr_str += self.short + "+" + self.longer
        repr_str += "\n"

        y_name = self.__shorter_var_name(reg_data["y_name"])
        repr_str += f" {y_name:>11} | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]"
        repr_str += "\n"
        repr_str += self.short + "+" + self.longer
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
            repr_str += f" {self.process_t_value(t_value[i]):>6}  "
            repr_str += f" {self.process_p_value(p_value[i]):>5}  "
            repr_str += f" {self.__shorter_float(ci_lower[i]):>9}  "
            repr_str += f" {self.__shorter_float(ci_upper[i]):>9}  "
            repr_str += "\n"
        repr_str += self.short + "+" + self.longer
        repr_str += "\n"
        return repr_str

    @staticmethod
    def process_t_value(value: float) -> str:
        """Format t-value to 2 decimal places.
        
        Args:
            value: The t-value to format.
            
        Returns:
            Formatted t-value string.
        """
        return f"{value:.2f}"

    @staticmethod
    def process_p_value(value: float) -> str:
        """Format p-value to 3 decimal places.
        
        Args:
            value: The p-value to format.
            
        Returns:
            Formatted p-value string.
        """
        return f"{value:.3f}"

    @staticmethod
    def __shorter_var_name(var_name: str, max_length: int = 11) -> str:
        """Shorten variable name if it exceeds maximum length.
        
        Args:
            var_name: The variable name to shorten.
            max_length: Maximum length allowed.
            
        Returns:
            Shortened variable name.
        """
        if len(var_name) <= max_length:
            var_name = f"{var_name:>11}"
            return var_name

        prefix = var_name[:5]
        suffix = var_name[-5:]
        return f"{prefix}~{suffix}"

    @staticmethod
    def __shorter_float(data: int | float, aim_length: int = 9) -> str:
        """Format float number to fit within specified length.
        
        Args:
            data: The number to format.
            aim_length: Target string length.
            
        Returns:
            Formatted number string that fits within the specified length.
        """
        # Handle sign
        sign = ' ' if data >= 0 else '-'
        abs_data = abs(data)

        # Special case: zero
        if abs_data == 0:
            result = sign + '0'
            return result.rjust(aim_length)

        # Determine if scientific notation is needed
        # For very large or very small numbers, use scientific notation
        if abs_data >= 1:
            # Large numbers: if integer part has too many digits, use scientific notation
            integer_digits = len(str(int(abs_data)))
            if integer_digits > aim_length - 2:  # Subtract sign and decimal point
                # Use scientific notation
                formatted = f"{abs_data:.2e}"
                result = sign + formatted
            else:
                # Regular format, calculate available decimal places
                available_decimals = aim_length - 1 - integer_digits - 1  # sign-integer-decimal point
                if available_decimals > 0:
                    result = sign + f"{abs_data:.{available_decimals}f}".rstrip('0').rstrip('.')
                else:
                    result = sign + str(int(abs_data))
        else:
            # Small decimals: count leading zeros
            str_data = f"{abs_data:.10f}"  # Start with sufficient precision
            decimal_part = str_data.split('.')[1]

            # Count leading zeros
            leading_zeros = 0
            for char in decimal_part:
                if char == '0':
                    leading_zeros += 1
                else:
                    break

            # If too many leading zeros (e.g., >= 4), use scientific notation
            if leading_zeros >= 4:
                formatted = f"{abs_data:.2e}"
                result = sign + formatted
            else:
                # Regular decimal format, remove leading zeros
                available_decimals = aim_length - 2  # sign + decimal point (removed leading zero)
                formatted_num = f"{abs_data:.{available_decimals}f}".rstrip('0').rstrip('.')
                # Remove "0." prefix, keep only the part after decimal point
                if formatted_num.startswith('0.'):
                    formatted_num = '.' + formatted_num[2:]
                result = sign + formatted_num

        # If result is too long, force scientific notation
        if len(result) > aim_length:
            formatted = f"{abs_data:.2e}"
            result = sign + formatted

        # Right align
        return result.rjust(aim_length)