import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Processes raw NBA game data for model training.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def calculate_fantasy_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate fantasy points based on game statistics.
        
        FP = PTS + 1.2*REB + 1.5*AST + 3*STL + 3*BLK - TOV
        
        Args:
            df: DataFrame with game statistics
            
        Returns:
            DataFrame with added fantasy points column
        """
        df = df.copy()
        
        try:
            df['FANTASY_POINTS'] = (
                df['PTS'] +
                1.2 * df['REB'] +
                1.5 * df['AST'] +
                3.0 * df['STL'] +
                3.0 * df['BLK'] -
                df['TOV']
            )
            return df
        except KeyError as e:
            logger.error(f"Missing required column for FP calculation: {str(e)}")
            raise
            
    def remove_outliers(self, 
                       df: pd.DataFrame,
                       columns: List[str],
                       std_threshold: float = 3.0) -> pd.DataFrame:
        """
        Remove statistical outliers from specified columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to check for outliers
            std_threshold: Number of standard deviations for outlier cutoff
            
        Returns:
            DataFrame with outliers removed
        """
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                df = df[
                    (df[col] >= mean - std_threshold * std) &
                    (df[col] <= mean + std_threshold * std)
                ]
                
        return df
        
    def handle_missing_values(self,
                            df: pd.DataFrame,
                            strategy: str = 'mean') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            strategy: Strategy for filling missing values ('mean', 'median', 'zero')
            
        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        
        if strategy == 'mean':
            df = df.fillna(df.mean())
        elif strategy == 'median':
            df = df.fillna(df.median())
        elif strategy == 'zero':
            df = df.fillna(0)
        else:
            raise ValueError(f"Unknown missing value strategy: {strategy}")
            
        return df
        
    def scale_features(self,
                      df: pd.DataFrame,
                      columns: List[str],
                      fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            df: Input DataFrame
            columns: Columns to scale
            fit: Whether to fit the scaler or use pre-fitted
            
        Returns:
            DataFrame with scaled features
        """
        df = df.copy()
        
        if fit:
            df[columns] = self.scaler.fit_transform(df[columns])
        else:
            df[columns] = self.scaler.transform(df[columns])
            
        return df
        
    def process_game_data(self,
                         df: pd.DataFrame,
                         scale: bool = True,
                         remove_outliers: bool = True) -> pd.DataFrame:
        """
        Complete data processing pipeline.
        
        Args:
            df: Raw game data
            scale: Whether to scale features
            remove_outliers: Whether to remove statistical outliers
            
        Returns:
            Processed DataFrame ready for feature engineering
        """
        try:
            # Calculate fantasy points
            df = self.calculate_fantasy_points(df)
            
            # Handle missing values
            df = self.handle_missing_values(df)
            
            # Remove outliers if requested
            if remove_outliers:
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                df = self.remove_outliers(df, numerical_cols.tolist())
            
            # Scale features if requested
            if scale:
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                df = self.scale_features(df, numerical_cols.tolist())
            
            return df
            
        except Exception as e:
            logger.error(f"Error in data processing pipeline: {str(e)}")
            raise