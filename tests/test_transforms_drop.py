import pandas as pd
import numpy as np
import pytest

from deepecon.transforms.drop import DropVar, KeepVar, DropCondition, KeepCondition


def simu_data() -> pd.DataFrame:
    """Generate a simulated dataset for testing drop operations"""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'age': [20, 25, 30, 35, 40],
        'income': [50000, 60000, 70000, 80000, 90000],
        'score': [85, 90, 78, 92, 88],
        'status': ['active', 'inactive', 'active', 'active', 'inactive']
    })


class TestDropVar:
    def test_drop_single_column(self):
        """Test dropping a single column"""
        df = simu_data()
        drop_var = DropVar(df)
        result = drop_var(X_cols=["income"])
        
        expected = df.drop(columns=["income"])
        pd.testing.assert_frame_equal(result, expected)

    def test_drop_multiple_columns(self):
        """Test dropping multiple columns"""
        df = simu_data()
        drop_var = DropVar(df)
        result = drop_var(X_cols=["income", "score"])
        
        expected = df.drop(columns=["income", "score"])
        pd.testing.assert_frame_equal(result, expected)

    def test_drop_nonexistent_column(self):
        """Test error when dropping non-existent column"""
        df = simu_data()
        drop_var = DropVar(df)
        
        with pytest.raises(Exception):  # VarNotFoundError
            drop_var(X_cols=["nonexistent_column"])

    def test_drop_empty_list(self):
        """Test error when no columns provided"""
        df = simu_data()
        drop_var = DropVar(df)
        
        with pytest.raises(ValueError, match="Missing the vars to drop"):
            drop_var(X_cols=[])

    def test_drop_none(self):
        """Test error when X_cols is None"""
        df = simu_data()
        drop_var = DropVar(df)
        
        with pytest.raises(ValueError, match="Missing the vars to drop"):
            drop_var(X_cols=None)


class TestKeepVar:
    def test_keep_single_column(self):
        """Test keeping a single column"""
        df = simu_data()
        keep_var = KeepVar(df)
        result = keep_var(X_cols=["age"])
        
        expected = df[["age"]]
        pd.testing.assert_frame_equal(result, expected)

    def test_keep_multiple_columns(self):
        """Test keeping multiple columns"""
        df = simu_data()
        keep_var = KeepVar(df)
        result = keep_var(X_cols=["age", "income"])
        
        expected = df[["age", "income"]]
        pd.testing.assert_frame_equal(result, expected)

    def test_keep_nonexistent_column(self):
        """Test error when keeping non-existent column"""
        df = simu_data()
        keep_var = KeepVar(df)
        
        with pytest.raises(Exception):  # VarNotFoundError
            keep_var(X_cols=["nonexistent_column"])

    def test_keep_empty_list(self):
        """Test error when no columns provided"""
        df = simu_data()
        keep_var = KeepVar(df)
        
        with pytest.raises(ValueError, match="Missing the vars to keep"):
            keep_var(X_cols=[])

    def test_keep_none(self):
        """Test error when X_cols is None"""
        df = simu_data()
        keep_var = KeepVar(df)
        
        with pytest.raises(ValueError, match="Missing the vars to keep"):
            keep_var(X_cols=None)


class TestDropCondition:
    def test_drop_conditional_rows(self):
        """Test dropping rows based on condition"""
        df = simu_data()
        drop_condition = DropCondition(df)
        
        # Create condition function
        def age_condition(df):
            return df['age'] > 30
        
        result = drop_condition(_if_exp=age_condition)
        expected = df[~(df['age'] > 30)]
        pd.testing.assert_frame_equal(result, expected)

    def test_drop_with_multiple_conditions(self):
        """Test dropping rows with complex condition"""
        df = simu_data()
        drop_condition = DropCondition(df)
        
        # Create complex condition function
        def complex_condition(df):
            return (df['income'] > 70000) & (df['status'] == 'active')
        
        result = drop_condition(_if_exp=complex_condition)
        expected = df[~((df['income'] > 70000) & (df['status'] == 'active'))]
        pd.testing.assert_frame_equal(result, expected)

    def test_drop_no_rows(self):
        """Test when condition doesn't match any rows"""
        df = simu_data()
        drop_condition = DropCondition(df)
        
        # Create condition that matches no rows
        def no_match_condition(df):
            return df['age'] > 100
        
        result = drop_condition(_if_exp=no_match_condition)
        expected = df.copy()
        pd.testing.assert_frame_equal(result, expected)

    def test_drop_all_rows(self):
        """Test when condition matches all rows"""
        df = simu_data()
        drop_condition = DropCondition(df)
        
        # Create condition that matches all rows
        def all_match_condition(df):
            return df['age'] > 0
        
        result = drop_condition(_if_exp=all_match_condition)
        expected = df[~(df['age'] > 0)]
        pd.testing.assert_frame_equal(result, expected)

    def test_drop_condition_none(self):
        """Test error when condition is None"""
        df = simu_data()
        drop_condition = DropCondition(df)
        
        with pytest.raises(Exception):  # ConditionNotFoundError
            drop_condition(_if_exp=None)


class TestKeepCondition:
    def test_keep_conditional_rows(self):
        """Test keeping rows based on condition"""
        df = simu_data()
        keep_condition = KeepCondition(df)
        
        # Create condition: age > 30
        def age_condition(df):
            return df['age'] > 30
        
        result = keep_condition(_if_exp=age_condition)
        expected = df[df['age'] > 30]
        pd.testing.assert_frame_equal(result, expected)

    def test_keep_with_multiple_conditions(self):
        """Test keeping rows with complex condition"""
        df = simu_data()
        keep_condition = KeepCondition(df)
        
        # Create condition: score >= 90 and income > 60000
        def complex_condition(df):
            return (df['score'] >= 90) & (df['income'] > 60000)
        
        result = keep_condition(_if_exp=complex_condition)
        expected = df[(df['score'] >= 90) & (df['income'] > 60000)]
        pd.testing.assert_frame_equal(result, expected)

    def test_keep_no_rows(self):
        """Test when condition doesn't match any rows"""
        df = simu_data()
        keep_condition = KeepCondition(df)
        
        # Create condition that matches no rows
        def no_match_condition(df):
            return df['age'] > 100
        
        result = keep_condition(_if_exp=no_match_condition)
        expected = df[df['age'] > 100]
        pd.testing.assert_frame_equal(result, expected)

    def test_keep_all_rows(self):
        """Test when condition matches all rows"""
        df = simu_data()
        keep_condition = KeepCondition(df)
        
        # Create condition that matches all rows
        def all_match_condition(df):
            return df['age'] > 0
        
        result = keep_condition(_if_exp=all_match_condition)
        expected = df[df['age'] > 0]
        pd.testing.assert_frame_equal(result, expected)

    def test_keep_condition_none(self):
        """Test error when condition is None"""
        df = simu_data()
        keep_condition = KeepCondition(df)
        
        with pytest.raises(Exception):  # ConditionNotFoundError
            keep_condition(_if_exp=None)


class TestIntegration:
    def test_chained_operations(self):
        """Test chaining drop operations"""
        df = simu_data()
        
        # First drop some columns
        drop_var = DropVar(df)
        intermediate = drop_var(X_cols=["score", "status"])
        
        # Then drop rows based on condition
        drop_condition = DropCondition(intermediate)
        def age_condition(df):
            return df['age'] > 30
        final = drop_condition(_if_exp=age_condition)
        
        expected = df.drop(columns=["score", "status"])
        expected = expected[expected['age'] <= 30]
        pd.testing.assert_frame_equal(final, expected)

    def test_keep_and_drop_combination(self):
        """Test combination of keep and drop operations"""
        df = simu_data()
        
        # First keep only certain columns
        keep_var = KeepVar(df)
        intermediate = keep_var(X_cols=["id", "age", "income"])
        
        # Then drop rows based on condition
        drop_condition = DropCondition(intermediate)
        def income_condition(df):
            return df["income"] > 70000
        final = drop_condition(_if_exp=income_condition)
        
        expected = df[["id", "age", "income"]]
        expected = expected[expected["income"] <= 70000]
        pd.testing.assert_frame_equal(final, expected)

