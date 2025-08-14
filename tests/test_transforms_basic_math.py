import pandas as pd
import numpy as np
import pytest
from deepecon.transforms.basic import BasicMath


def simu_data() -> pd.DataFrame:
    """Generate a simulated dataset with age column and three x columns (x1, x2, x3), 
    where x3 contains zeros and negative values"""
    np.random.seed(42)
    return pd.DataFrame({
        'age': [20, 25, 30, 35, 40, 45, 50],
        'x1': [1, 2, 3, 4, 5, 6, 7],
        'x2': [10, 20, 30, 40, 50, 60, 70],
        'x3': [1, 0, -1, 2, 0, -2, 3]  # Contains zeros and negative values
    })


class TestBasicMath:
    def test_power_operation(self):
        """Test power operation: gen age2 = age ^ 2"""
        df = simu_data()
        basic_math = BasicMath(df)
        result = basic_math(y_cols=["age2"], op="age ^ 2")
        
        expected = df['age'] ** 2
        pd.testing.assert_series_equal(result['age2'], expected, check_names=False)

    def test_addition_operation(self):
        """Test addition operation: gen y1 = x1 + 1"""
        df = simu_data()
        basic_math = BasicMath(df)
        result = basic_math(y_cols=["y1"], op="x1 + 1")
        
        expected = df['x1'] + 1
        pd.testing.assert_series_equal(result['y1'], expected, check_names=False)

    def test_replace_operation(self):
        """Test replace operation: replace age = age + 1"""
        df = simu_data()
        original_age = df['age'].copy()
        basic_math = BasicMath(df)
        result = basic_math(y_cols=["age"], op="age + 1", replace=True)
        
        expected = original_age + 1
        pd.testing.assert_series_equal(result['age'], expected)

    def test_complex_operation(self):
        """Test complex operation: gen y1 = x1 * x2 / 2 + log(age)"""
        df = simu_data()
        basic_math = BasicMath(df)
        result = basic_math(y_cols=["y1"], op="x1 * x2 / 2 + log(age)")
        
        expected = (df['x1'] * df['x2'] / 2) + np.log(df['age'])
        pd.testing.assert_series_equal(result['y1'], expected, check_names=False)

    def test_division_by_zero(self):
        """Test division by zero: gen y1 = x1 / x3"""
        df = simu_data()
        basic_math = BasicMath(df)
        result = basic_math(y_cols=["y1"], op="x1 / x3")
        
        expected = df['x1'] / df['x3']
        # Allow inf values as they are mathematically correct for division by zero
        pd.testing.assert_series_equal(result['y1'], expected, check_names=False)

    def test_log_negative_values(self):
        """Test log with negative values: gen y1 = log(x3)"""
        df = simu_data()
        basic_math = BasicMath(df)
        result = basic_math(y_cols=["y1"], op="log(x3)")
        
        expected = np.log(df['x3'])
        # Allow actual numpy behavior for log of negative values and zero
        pd.testing.assert_series_equal(result['y1'], expected, check_names=False)

    def test_multi_var_operation(self):
        """Test multi-variable operation: gen y1 = x1 * x2 + x3"""
        df = simu_data()
        basic_math = BasicMath(df)
        result = basic_math(y_cols=["y1"], op="x1 * x2 + x3")
        
        expected = df['x1'] * df['x2'] + df['x3']
        pd.testing.assert_series_equal(result['y1'], expected, check_names=False)

    def test_constant_operation(self):
        """Test operation with constant: gen y1 = 5 + 3"""
        df = simu_data()
        basic_math = BasicMath(df)
        result = basic_math(y_cols=["y1"], op="5 + 3")
        
        expected = pd.Series([8] * len(df))
        pd.testing.assert_series_equal(result['y1'], expected, check_names=False)

    def test_invalid_variable_error(self):
        """Test error handling for invalid variable"""
        df = simu_data()
        basic_math = BasicMath(df)
        
        with pytest.raises(Exception):  # Accept any exception type
            basic_math(y_cols=["y1"], op="nonexistent_column + 1")

    def test_column_already_exists_error(self):
        """Test error handling for existing column without replace"""
        df = simu_data()
        basic_math = BasicMath(df)
        
        with pytest.raises(ValueError, match="already exists"):
            basic_math(y_cols=["age"], op="age + 1", replace=False)

    def test_exponentiation_operation(self):
        """Test exponentiation: gen y1 = exp(x1)"""
        df = simu_data()
        basic_math = BasicMath(df)
        result = basic_math(y_cols=["y1"], op="exp(x1)")
        
        expected = np.exp(df['x1'])
        pd.testing.assert_series_equal(result['y1'], expected, check_names=False)

    def test_sqrt_operation(self):
        """Test square root: gen y1 = sqrt(x1)"""
        df = simu_data()
        basic_math = BasicMath(df)
        result = basic_math(y_cols=["y1"], op="sqrt(x1)")
        
        expected = np.sqrt(df['x1'])
        pd.testing.assert_series_equal(result['y1'], expected, check_names=False)

    def test_nested_operations(self):
        """Test nested operations: gen y1 = (x1 + x2) * (x3 - 1)"""
        df = simu_data()
        basic_math = BasicMath(df)
        result = basic_math(y_cols=["y1"], op="(x1 + x2) * (x3 - 1)")
        
        expected = (df['x1'] + df['x2']) * (df['x3'] - 1)
        pd.testing.assert_series_equal(result['y1'], expected, check_names=False)
