import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from scipy.stats import yeojohnson
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureBuilder:
    """
    Builds features for NBA player performance prediction.
    """
    
    def __init__(self):
        self.feature_columns = []
        
    def create_lag_features(self,
                           df: pd.DataFrame,
                           columns: List[str],
                           lags: List[int]) -> pd.DataFrame:
        """
        Create lagged features for specified columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with added lag features
        """
        df = df.copy()
        
        try:
            for col in columns:
                for lag in lags:
                    lag_col = f"{col}_lag_{lag}"
                    df[lag_col] = df[col].shift(lag)
                    self.feature_columns.append(lag_col)
                    
            return df
            
        except Exception as e:
            logger.error(f"Error creating lag features: {str(e)}")
            raise
            
    def create_rolling_features(self,
                              df: pd.DataFrame,
                              columns: List[str],
                              windows: List[int]) -> pd.DataFrame:
        """
        Create rolling average features.
        
        Args:
            df: Input DataFrame
            columns: Columns to create rolling features for
            windows: List of rolling window sizes
            
        Returns:
            DataFrame with added rolling features
        """
        df = df.copy()
        
        try:
            for col in columns:
                for window in windows:
                    # Rolling mean
                    mean_col = f"{col}_rolling_mean_{window}"
                    df[mean_col] = df[col].rolling(window=window).mean()
                    self.feature_columns.append(mean_col)
                    
                    # Rolling std
                    std_col = f"{col}_rolling_std_{window}"
                    df[std_col] = df[col].rolling(window=window).std()
                    self.feature_columns.append(std_col)
                    
            return df
            
        except Exception as e:
            logger.error(f"Error creating rolling features: {str(e)}")
            raise
            
    def create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create momentum indicators.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with added momentum features
        """
        df = df.copy()
        
        try:
            # Game-to-game changes
            for col in ['FANTASY_POINTS', 'PTS', 'REB', 'AST']:
                momentum_col = f"{col}_momentum"
                df[momentum_col] = df[col].diff()
                self.feature_columns.append(momentum_col)
                
            # Trend indicators (positive/negative streaks)
            for col in ['FANTASY_POINTS']:
                streak_col = f"{col}_streak"
                df[streak_col] = (df[col].diff() > 0).astype(int)
                df[streak_col] = df[streak_col].groupby(
                    (df[streak_col] != df[streak_col].shift()).cumsum()
                ).cumsum()
                self.feature_columns.append(streak_col)
                
            return df
            
        except Exception as e:
            logger.error(f"Error creating momentum features: {str(e)}")
            raise
            
    def create_game_situation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features related to game situation.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with added game situation features
        """
        df = df.copy()
        
        try:
            # Home/Away encoding
            df['IS_HOME'] = (df['MATCHUP'].str.contains('vs')).astype(int)
            self.feature_columns.append('IS_HOME')
            
            # Days rest
            df['DAYS_REST'] = df['GAME_DATE'].diff().dt.days
            df['DAYS_REST'] = df['DAYS_REST'].fillna(3)  # Average rest for first game
            self.feature_columns.append('DAYS_REST')
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating game situation features: {str(e)}")
            raise
            
    def transform_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Yeo-Johnson transformation to target variable.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with transformed target
        """
        df = df.copy()
        
        try:
            df['FANTASY_POINTS_TRANSFORMED'], _ = yeojohnson(df['FANTASY_POINTS'])
            return df
        except Exception as e: