import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def sample_game_data():
    """
    Create sample game data for testing.
    """
    return pd.DataFrame({
        'GAME_DATE': [
            datetime(2024, 1, 1),
            datetime(2024, 1, 3),
            datetime(2024, 1, 5)
        ],
        'PTS': [25, 30, 28],
        'REB': [5, 8, 6],
        'AST': [7, 4, 9],
        'STL': [2, 1, 3],
        'BLK': [1, 2, 0],
        'TOV': [3, 2, 4],
        'MIN': [34, 36, 32],
        'MATCHUP': ['vs GSW', '@ LAL', 'vs BOS']
    })

@pytest.fixture
def sample_processed_data():
    """
    Create sample processed data with fantasy points.
    """
    df = pd.DataFrame({
        'GAME_DATE': pd.date_range(start='2024-01-01', periods=10),
        'PTS': np.random.randint(10, 40, 10),
        'REB': np.random.randint(2, 15, 10),
        'AST': np.random.randint(1, 12, 10),
        'STL': np.random.randint(0, 5, 10),
        'BLK': np.random.randint(0, 4, 10),
        'TOV': np.random.randint(1, 6, 10),
        'MIN': np.random.randint(20, 40, 10)
    })
    
    # Calculate fantasy points
    df['FANTASY_POINTS'] = (
        df['PTS'] +
        1.2 * df['REB'] +
        1.5 * df['AST'] +
        3.0 * df['STL'] +
        3.0 * df['BLK'] -
        df['TOV']
    )
    
    return df

@pytest.fixture
def sample_features_data():
    """
    Create sample data with engineered features.
    """
    df = pd.DataFrame({
        'GAME_DATE': pd.date_range(start='2024-01-01', periods=15),
        'FANTASY_POINTS': np.random.normal(35, 5, 15),
        'PTS': np.random.normal(25, 5, 15),
        'REB': np.random.normal(8, 2, 15),
        'AST': np.random.normal(6, 2, 15),
        'MIN': np.random.normal(32, 3, 15),
        'MATCHUP': ['vs ' + team for team in np.random.choice(['GSW', 'LAL', 'BOS', 'PHI', 'MIA'], 15)]
    })
    return df

@pytest.fixture
def sample_train_data():
    """
    Create sample training data.
    """
    X = pd.DataFrame({
        'PTS_rolling_3': np.random.normal(25, 5, 100),
        'REB_rolling_3': np.random.normal(8, 2, 100),
        'AST_rolling_3': np.random.normal(6, 2, 100),
        'FANTASY_POINTS_momentum': np.random.normal(0, 5, 100),
        'MIN_rolling_5': np.random.normal(32, 3, 100)
    })
    
    y = np.random.normal(35, 5, 100)
    
    return X, y