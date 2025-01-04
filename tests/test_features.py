import pytest
import pandas as pd
import numpy as np
from src.features.builder import FeatureBuilder

def test_create_lag_features(sample_features_data):
    """Test creation of lag features."""
    builder = FeatureBuilder()
    
    result = builder.create_lag_features(
        sample_features_data,
        columns=['FANTASY_POINTS', 'PTS'],
        lags=[1, 3, 5]
    )
    
    # Check that lag features were created
    assert 'FANTASY_POINTS_lag_1' in result.columns
    assert 'FANTASY_POINTS_lag_3' in result.columns
    assert 'FANTASY_POINTS_lag_5' in result.columns
    assert 'PTS_lag_1' in result.columns
    assert 'PTS_lag_3' in result.columns
    assert 'PTS_lag_5' in result.columns
    
    # Check that lag values are correct
    assert result['FANTASY_POINTS_lag_1'].iloc[1] == sample_features_data['FANTASY_POINTS'].iloc[0]

def test_create_rolling_features(sample_features_data):
    """Test creation of rolling features."""
    builder = FeatureBuilder()
    
    result = builder.create_rolling_features(
        sample_features_data,
        columns=['FANTASY_POINTS', 'PTS'],
        windows=[3, 5]
    )
    
    # Check that rolling features were created
    expected_columns = [
        'FANTASY_POINTS_rolling_mean_3',
        'FANTASY_POINTS_rolling_std_3',
        'FANTASY_POINTS_rolling_mean_5',
        'FANTASY_POINTS_rolling_std_5',
        'PTS_rolling_mean_3',
        'PTS_rolling_std_3',
        'PTS_rolling_mean_5',
        'PTS_rolling_std_5'
    ]
    
    for col in expected_columns:
        assert col in result.columns

def test_create_momentum_features(sample_features_data):
    """Test creation of momentum features."""
    builder = FeatureBuilder()
    
    result = builder.create_momentum