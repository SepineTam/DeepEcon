#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : test_transforms_corr.py

import numpy as np
import pandas as pd
import pytest

from deepecon.transforms.corr.base import CorrelationBase
from deepecon.transforms.corr.pearson import PearsonCorr
from deepecon.core.errors import LengthNotMatchError, VarNotFoundError


class TestCorrelationBase:
    """Test CorrelationBase base class"""
    
    @pytest.fixture
    def sample_df(self):
        """Create test DataFrame"""
        return pd.DataFrame({
            'x1': [1, 2, 3, 4, 5],
            'x2': [2, 4, 6, 8, 10],
            'x3': [1, 3, 5, 7, 9],
            'y': [1.5, 3.0, 4.5, 6.0, 7.5]
        })
    
    @pytest.fixture
    def correlation_base(self, sample_df):
        """Create CorrelationBase instance - using PearsonCorr as concrete implementation"""
        return PearsonCorr(sample_df)
    
    def test_options(self, correlation_base):
        """Test options method"""
        options = correlation_base.options()
        assert "X_cols" in options
        assert isinstance(options, dict)
    
    def test_array_corr_method(self, correlation_base):
        """Test array_corr method"""
        X_cols = ['x1', 'x2', 'x3']
        result = correlation_base.array_corr(X_cols)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 3)
        assert list(result.index) == X_cols
        assert list(result.columns) == X_cols
        
        # Diagonal should all be 1
        for col in X_cols:
            assert result.loc[col, col] == 1.0
    
    def test_y_x_corr_method(self, correlation_base):
        """Test y_x_corr method"""
        y_col = 'y'
        X_cols = ['x1', 'x2']
        result = correlation_base.y_x_corr(y_col, X_cols)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (1, 2)
        assert list(result.columns) == X_cols
    
    def test_transform_array_mode(self, correlation_base):
        """Test transform method array mode"""
        X_cols = ['x1', 'x2', 'x3']
        result = correlation_base.transform(
            X_cols=X_cols,
            is_array=True
        )
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 3)
    
    def test_transform_y_x_mode(self, correlation_base):
        """Test transform method y_x mode"""
        y_col = 'y'
        X_cols = ['x1', 'x2']
        result = correlation_base.transform(
            y_col=y_col,
            X_cols=X_cols
        )
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (1, 2)
    
    def test_transform_array_mode_insufficient_cols(self, correlation_base):
        """Test array mode insufficient columns error"""
        with pytest.raises(LengthNotMatchError):
            correlation_base.transform(
                X_cols=['x1'],
                is_array=True
            )
    
    def test_transform_y_x_mode_no_y_col(self, correlation_base):
        """Test y_x mode y_col not string error"""
        with pytest.raises(TypeError):
            correlation_base.transform(
                y_col=123,  # Not a string
                X_cols=['x1', 'x2']
            )
    
    def test_transform_y_x_mode_empty_x_cols(self, correlation_base):
        """Test y_x mode empty X_cols error"""
        with pytest.raises(LengthNotMatchError):
            correlation_base.transform(
                y_col='y',
                X_cols=[]
            )
    
    def test_pre_process_var_not_found(self, correlation_base):
        """Test pre_process method variable not found scenario"""
        with pytest.raises(VarNotFoundError):
            correlation_base.pre_process(['non_existent_col'])


class TestPearsonCorr:
    """Test PearsonCorr class"""
    
    @pytest.fixture
    def sample_df(self):
        """Create test DataFrame"""
        return pd.DataFrame({
            'x1': [1, 2, 3, 4, 5],
            'x2': [2, 4, 6, 8, 10],  # Perfect positive correlation with x1
            'x3': [-1, -2, -3, -4, -5],  # Perfect negative correlation with x1
            'x4': [1, 1, 1, 1, 1],  # Constant, no correlation
            'y': [1.5, 3.0, 4.5, 6.0, 7.5]
        })
    
    @pytest.fixture
    def pearson_corr(self, sample_df):
        """Create PearsonCorr instance"""
        return PearsonCorr(sample_df)
    
    def test_base_corr_perfect_positive(self, pearson_corr):
        """Test perfect positive correlation coefficient"""
        result = pearson_corr._base_corr('x1', 'x2')
        assert abs(result - 1.0) < 1e-10
    
    def test_base_corr_perfect_negative(self, pearson_corr):
        """Test perfect negative correlation coefficient"""
        result = pearson_corr._base_corr('x1', 'x3')
        assert abs(result - (-1.0)) < 1e-10
    
    def test_base_corr_no_correlation(self, pearson_corr):
        """Test no correlation scenario"""
        result = pearson_corr._base_corr('x1', 'x4')
        assert np.isnan(result)
    
    def test_base_corr_with_nan_values(self, pearson_corr):
        """Test correlation with NaN values"""
        # Add some NaN values
        pearson_corr.df.loc[2, 'x1'] = np.nan
        result = pearson_corr._base_corr('x1', 'x2')
        # Should still be able to calculate correlation
        assert not np.isnan(result)
    
    def test_base_corr_insufficient_data(self, pearson_corr):
        """Test insufficient data scenario"""
        # Only keep one row of data
        pearson_corr.df = pearson_corr.df.iloc[:1]
        result = pearson_corr._base_corr('x1', 'x2')
        assert np.isnan(result)
    
    def test_transform_array_mode(self, pearson_corr):
        """Test transform method array mode"""
        X_cols = ['x1', 'x2', 'x3']
        result = pearson_corr.transform(
            X_cols=X_cols,
            is_array=True
        )
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 3)
        
        # Check diagonal
        for col in X_cols:
            assert result.loc[col, col] == 1.0
        
        # Check symmetry
        assert abs(result.loc['x1', 'x2'] - result.loc['x2', 'x1']) < 1e-10
        assert abs(result.loc['x1', 'x3'] - result.loc['x3', 'x1']) < 1e-10
    
    def test_transform_y_x_mode(self, pearson_corr):
        """Test transform method y_x mode"""
        y_col = 'y'
        X_cols = ['x1', 'x2']
        result = pearson_corr.transform(
            y_col=y_col,
            X_cols=X_cols
        )
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (1, 2)
        assert list(result.columns) == X_cols
        
        # Check correlation coefficients are within reasonable range
        for col in X_cols:
            corr_value = result.iloc[0][col]
            assert -1.0 <= corr_value <= 1.0
    
    def test_options(self, pearson_corr):
        """Test options method"""
        options = pearson_corr.options()
        assert "X_cols" in options
        assert isinstance(options, dict)


class TestCorrelationEdgeCases:
    """Test correlation edge cases"""
    
    @pytest.fixture
    def edge_case_df(self):
        """Create DataFrame for edge case testing"""
        return pd.DataFrame({
            'zero_var': [0, 0, 0, 0, 0],  # Zero variance
            'constant': [5, 5, 5, 5, 5],   # Constant
            'mixed': [1, 2, np.nan, 4, 5], # Contains NaN
            'normal': [1, 2, 3, 4, 5]      # Normal data
        })
    
    @pytest.fixture
    def pearson_corr(self, edge_case_df):
        """Create PearsonCorr instance"""
        return PearsonCorr(edge_case_df)
    
    def test_zero_variance_correlation(self, pearson_corr):
        """Test correlation for zero variance variables"""
        result = pearson_corr._base_corr('zero_var', 'normal')
        assert np.isnan(result)
    
    def test_constant_variable_correlation(self, pearson_corr):
        """Test correlation for constant variables"""
        result = pearson_corr._base_corr('constant', 'normal')
        assert np.isnan(result)
    
    def test_mixed_data_correlation(self, pearson_corr):
        """Test correlation for mixed data (contains NaN)"""
        result = pearson_corr._base_corr('mixed', 'normal')
        # Should be able to calculate correlation, may not be NaN
        assert isinstance(result, float)
    
    def test_self_correlation(self, pearson_corr):
        """Test variable correlation with itself"""
        result = pearson_corr._base_corr('normal', 'normal')
        assert abs(result - 1.0) < 1e-10


class TestCorrelationIntegration:
    """Test correlation integration"""
    
    @pytest.fixture
    def large_df(self):
        """Create large test dataset"""
        np.random.seed(42)
        n = 1000
        x = np.random.normal(0, 1, n)
        y = 0.7 * x + np.random.normal(0, 0.5, n)  # Correlation coefficient approximately 0.7
        z = -0.3 * x + np.random.normal(0, 0.8, n)  # Correlation coefficient approximately -0.3
        
        return pd.DataFrame({
            'x': x,
            'y': y,
            'z': z
        })
    
    @pytest.fixture
    def pearson_corr(self, large_df):
        """Create PearsonCorr instance"""
        return PearsonCorr(large_df)
    
    def test_large_dataset_correlation(self, pearson_corr):
        """Test large dataset correlation calculation"""
        result = pearson_corr._base_corr('x', 'y')
        
        # Correlation coefficient should be within reasonable range
        assert -1.0 <= result <= 1.0
        # Should be close to 0.7, but allow larger margin of error
        assert abs(result - 0.7) < 0.15
    
    def test_large_dataset_array_corr(self, pearson_corr):
        """Test large dataset array correlation"""
        X_cols = ['x', 'y', 'z']
        result = pearson_corr.array_corr(X_cols)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 3)
        
        # Check symmetry
        assert abs(result.loc['x', 'y'] - result.loc['y', 'x']) < 1e-10
        assert abs(result.loc['x', 'z'] - result.loc['z', 'x']) < 1e-10
        assert abs(result.loc['y', 'z'] - result.loc['z', 'y']) < 1e-10
    
    def test_large_dataset_y_x_corr(self, pearson_corr):
        """Test large dataset y_x correlation"""
        y_col = 'x'
        X_cols = ['y', 'z']
        result = pearson_corr.y_x_corr(y_col, X_cols)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (1, 2)
        assert list(result.columns) == X_cols
