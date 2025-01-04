import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from sklearn.base import BaseEstimator
import joblib
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Predictor:
    """
    Makes predictions for NBA player performance using trained models.
    """
    
    def __init__(self, model: BaseEstimator = None):
        """
        Initialize predictor with optional model.
        
        Args:
            model: Pre-trained model (optional)
        """
        self.model = model
        
    def load_model(self, model_path: str):
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to saved model file
        """
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
            
    def validate_input_data(self, df: pd.DataFrame, required_features: List[str]) -> bool:
        """
        Validate that input data has required features.
        
        Args:
            df: Input DataFrame
            required_features: List of required feature names
            
        Returns:
            Boolean indicating if data is valid
        """
        missing_features = [f for f in required_features if f not in df.columns]
        
        if missing_features:
            logger.error(f"Missing required features: {missing_features}")
            return False
        return True
        
    def prepare_prediction_data(self, 
                              df: pd.DataFrame,
                              required_features: List[str]) -> pd.DataFrame:
        """
        Prepare data for prediction.
        
        Args:
            df: Input DataFrame
            required_features: List of required feature names
            
        Returns:
            DataFrame ready for prediction
        """
        # Validate input data
        if not self.validate_input_data(df, required_features):
            raise ValueError("Invalid input data")
            
        # Select required features
        X = df[required_features].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        return X
        
    def predict(self, 
                X: pd.DataFrame,
                required_features: Optional[List[str]] = None) -> np.ndarray:
        """
        Make predictions using trained model.
        
        Args:
            X: Input features
            required_features: List of required feature names (optional)
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("No model loaded. Load model before prediction.")
            
        try:
            # Prepare data if required features specified
            if required_features is not None:
                X = self.prepare_prediction_data(X, required_features)
                
            # Make predictions
            predictions = self.model.predict(X)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
            
    def predict_with_confidence(self,
                              X: pd.DataFrame,
                              required_features: Optional[List[str]] = None,
                              n_bootstrap: int = 100) -> Dict[str, np.ndarray]:
        """
        Make predictions with confidence intervals using bootstrapping.
        
        Args:
            X: Input features
            required_features: List of required feature names (optional)
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Dictionary with predictions and confidence intervals
        """
        if self.model is None:
            raise ValueError("No model loaded. Load model before prediction.")
            
        try:
            # Prepare data if required features specified
            if required_features is not None:
                X = self.prepare_prediction_data(X, required_features)
                
            # Generate bootstrap predictions
            bootstrap_predictions = []
            n_samples = len(X)
            
            for _ in range(n_bootstrap):
                # Bootstrap sample
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                X_bootstrap = X.iloc[indices]
                
                # Make predictions
                y_pred = self.model.predict(X_bootstrap)
                bootstrap_predictions.append(y_pred)
                
            bootstrap_predictions = np.array(bootstrap_predictions)
            
            # Calculate mean and confidence intervals
            mean_pred = np.mean(bootstrap_predictions, axis=0)
            lower_ci = np.percentile(bootstrap_predictions, 2.5, axis=0)
            upper_ci = np.percentile(bootstrap_predictions, 97.5, axis=0)
            
            return {
                'predictions': mean_pred,
                'lower_ci': lower_ci,
                'upper_ci': upper_ci
            }
            
        except Exception as e:
            logger.error(f"Error making predictions with confidence intervals: {str(e)}")
            raise
            
    def predict_upcoming_games(self,
                             player_data: pd.DataFrame,
                             future_games: pd.DataFrame,
                             required_features: List[str]) -> pd.DataFrame:
        """
        Make predictions for upcoming games.
        
        Args:
            player_data: Historical player data
            future_games: DataFrame with upcoming game information
            required_features: List of required features
            
        Returns:
            DataFrame with predictions for upcoming games
        """
        predictions = []
        
        try:
            for _, game in future_games.iterrows():
                # Prepare features for game
                game_features = self._prepare_game_features(
                    player_data,
                    game,
                    required_features
                )
                
                # Make prediction
                pred = self.predict(game_features, required_features)[0]
                
                predictions.append({
                    'game_date': game['GAME_DATE'],
                    'opponent': game['OPPONENT'],
                    'predicted_fp': pred
                })
                
            return pd.DataFrame(predictions)
            
        except Exception as e:
            logger.error(f"Error predicting upcoming games: {str(e)}")
            raise
            
    def _prepare_game_features(self,
                             player_data: pd.DataFrame,
                             game: pd.Series,
                             required_features: List[str]) -> pd.DataFrame:
        """
        Prepare features for a specific game prediction.
        
        Args:
            player_data: Historical player data
            game: Series with game information
            required_features: List of required features
            
        Returns:
            DataFrame with features for prediction
        """
        # Create feature set based on historical data
        features = pd.DataFrame([game])
        
        # Add derived features
        features = self._add_derived_features(features, player_data)
        
        # Ensure all required features are present
        missing_features = set(required_features) - set(features.columns)
        for feature in missing_features:
            features[feature] = 0
            
        return features[required_features]
        
    def _add_derived_features(self,
                            features: pd.DataFrame,
                            historical_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features based on historical data.
        
        Args:
            features: Base features DataFrame
            historical_data: Historical player data
            
        Returns:
            DataFrame with added derived features
        """
        # Add rolling averages
        for col in ['PTS', 'REB', 'AST', 'FANTASY_POINTS']:
            if col in historical_data.columns:
                features[f'{col}_rolling_3'] = historical_data[col].rolling(3).mean().iloc[-1]
                features[f'{col}_rolling_5'] = historical_data[col].rolling(5).mean().iloc[-1]
                
        # Add momentum features
        for col in ['FANTASY_POINTS']:
            if col in historical_data.columns:
                features[f'{col}_momentum'] = historical_data[col].diff().iloc[-1]
                
        return features