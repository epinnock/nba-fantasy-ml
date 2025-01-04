import pytest
import pandas as pd
import numpy as np
from src.data.processor import DataProcessor

def test_calculate_fantasy_points(sample_game_data):
    """Test fantasy points calculation."""
    processor = DataProcessor()
    result = processor.calculate_fantasy_points(sample_game_data)
    
    # Manually calculate expected fantasy points
    expected_fp = (
        sample_game_data['PTS'] +
        1.2 * sample_game_data['REB'] +
        1.5 * sample_game_data['AST'] +
        3.0 * sample_game_data['STL'] +
        3.0 * sample_game_data['BLK'] -
        sample_game_data['TOV']
    )
    
    pd.testing.assert_series_equal(
        result['FANTASY_POINTS'],
        expected_fp,
        check_names=False
    )

def test_remove_outliers(sample_processed_data):
    """Test outlier removal."""
    processor = DataProcessor()
    
    # Add some outliers
    sample_processed_data.loc[0, 'PTS'] = 100  # Extreme value
    sample_processed_data.loc[1, 'REB'] = 50   # Extreme value
    
    result = processor.remove_outliers(
        sample_processed_data,
        columns=['PTS', 'REB'],
        std_threshold=2.0
    )
    
    # Check that outliers were removed
    assert len(result) < len(sample_processed_data)
    assert result['PTS'].max() < 100
    assert result['REB'].max() < 50

def test_handle_missing_values():
    """Test missing value handling."""
    processor = DataProcessor()
    
    # Create data with missing values
    data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, 4, 5],
        'C': [1, 2, 3, np.nan, 5]
    })
    
    # Test mean strategy
    result_mean = processor.handle_missing_values(data, strategy='mean')
    assert not result_mean.isna().any().any()
    
    # Test median strategy
    result_median = processor.handle_missing_values(data, strategy='median')
    assert not result_median.isna().any().any()
    
    # Test zero strategy
    result_zero = processor.handle_missing_values(data, strategy='zero')
    assert not result_zero.isna().any().any()
    assert (result_zero.fillna(0) == result_zero).all().all()

def test_scale_features(sample_processed_data):
    """Test feature scaling."""
    processor = DataProcessor()
    
    columns_to_scale = ['PTS', 'REB', 'AST']
    result = processor.scale_features(
        sample_processed_data,
        columns=columns_to_scale,
        fit=True
    )
    
    # Check that scaled features have mean close to 0 and std close to 1
    for col in columns_to_scale:
        assert abs(result[col].mean()) < 0.0001
        assert abs(result[col].std() - 1.0) < 0.0001

def test_process_game_data(sample_game_data):
    """Test complete data processing pipeline."""
    processor = DataProcessor()
    
    result = processor.process_game_data(
        sample_game_data,
        scale=True,
        remove_outliers=True
    )
    
    # Check that fantasy points were calculated
    assert 'FANTASY_POINTS' in result.columns
    
    # Check that numerical features were scaled
    numerical_cols = result.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        assert abs(result[col].mean()) < 0.0001
        assert abs(result[col].std() - 1.0) < 0.0001

def test_error_handling():
    """Test error handling in processor."""
    processor = DataProcessor()
    
    # Test missing required columns
    invalid_data = pd.DataFrame({
        'GAME_DATE': ['2024-01-01'],
        'POINTS': [25]  # Wrong column name
    })
    
    with pytest.raises(KeyError):
        processor.calculate_fantasy_points(invalid_data)
        
    # Test invalid scaling
    with pytest.raises(ValueError):
        processor.scale_features(
            invalid_data,
            columns=['NONEXISTENT_COLUMN']
        )
        
    # Test invalid missing value strategy
    with pytest.raises(ValueError):
        processor.handle_missing_values(
            invalid_data,
            strategy='invalid_strategy'
        )