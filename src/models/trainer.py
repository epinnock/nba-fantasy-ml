import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, VotingRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.base import BaseEstimator
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Handles training of NBA player performance prediction models.
    Implements ensemble approach from the paper using:
    - Random Forest
    - Bayesian Ridge
    - AdaBoost
    - Elastic Net
    """
    
    def __init__(self, 
                 model_dir: str = "models",
                 random_state: int = 42):
        """
        Initialize trainer with model configurations.
        
        Args:
            model_dir: Directory to save trained models
            random_state: Random seed for reproducibility
        """
        self.model_dir = model_dir
        self.random_state = random_state
        self.best_model = None
        self.feature_importances_ = None

    def create_base_models(self) -> List[Tuple[str, BaseEstimator]]:
        """
        Create the base models for the ensemble.
        
        Returns:
            List of (name, model) tuples
        """
        models = [
            ('rf', RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state
            )),
            ('br', BayesianRidge()),
            ('ada', AdaBoostRegressor(
                random_state=self.random_state
            )),
            ('en', ElasticNet(
                random_state=self.random_state
            ))
        ]
        return models

    def create_voting_ensemble(self, models: List[Tuple[str, BaseEstimator]]) -> VotingRegressor:
        """
        Create voting ensemble from base models.
        
        Args:
            models: List of (name, model) tuples
            
        Returns:
            VotingRegressor ensemble model
        """
        return VotingRegressor(estimators=models)

    def evaluate_model(self, 
                      model: BaseEstimator,
                      X_test: pd.DataFrame,
                      y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance using metrics from paper.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = model.predict(X_test)
        
        return {
            'mae': mean_absolute_error(y_test, predictions),
            'mape': mean_absolute_percentage_error(y_test, predictions) * 100
        }

    def calculate_feature_importance(self, 
                                   model: BaseEstimator,
                                   feature_names: List[str]) -> pd.DataFrame:
        """
        Calculate feature importance scores.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance scores
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            logger.warning("Model doesn't provide feature importances")
            return None
            
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        return importance_df.sort_values('importance', ascending=False)

    def train_model(self,
                   X_train: pd.DataFrame,
                   y_train: pd.Series,
                   X_val: Optional[pd.DataFrame] = None,
                   y_val: Optional[pd.Series] = None) -> BaseEstimator:
        """
        Train the model using ensemble approach.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Trained model
        """
        try:
            # Create and train base models
            base_models = self.create_base_models()
            ensemble = self.create_voting_ensemble(base_models)
            
            logger.info("Training ensemble model...")
            ensemble.fit(X_train, y_train)
            
            # Calculate feature importance
            self.feature_importances_ = self.calculate_feature_importance(
                ensemble.estimators_[0],  # Use Random Forest for importance
                X_train.columns
            )
            
            # Evaluate if validation set provided
            if X_val is not None and y_val is not None:
                metrics = self.evaluate_model(ensemble, X_val, y_val)
                logger.info(f"Validation metrics: MAE={metrics['mae']:.2f}, MAPE={metrics['mape']:.2f}%")
                
            self.best_model = ensemble
            return ensemble
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise

    def save_model(self, model: BaseEstimator, model_name: str):
        """
        Save trained model to disk.
        
        Args:
            model: Trained model to save
            model_name: Name for saved model file
        """
        try:
            path = f"{self.model_dir}/{model_name}.joblib"
            joblib.dump(model, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, model_name: str) -> BaseEstimator:
        """
        Load trained model from disk.
        
        Args:
            model_name: Name of model to load
            
        Returns:
            Loaded model
        """
        try:
            path = f"{self.model_dir}/{model_name}.joblib"
            model = joblib.load(path)
            logger.info(f"Model loaded from {path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise