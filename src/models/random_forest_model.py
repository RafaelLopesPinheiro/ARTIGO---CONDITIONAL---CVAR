"""
Random Forest implementation for quantile forecasting.
Uses quantile regression forests.
"""

from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from typing import Dict, List
import logging
import warnings
from .base_model import BaseQuantileModel

logger = logging.getLogger(__name__)


class RandomForestQuantileModel(BaseQuantileModel):
    """Random Forest-based quantile forecasting."""
    
    def __init__(
        self, 
        quantiles: List[float] = [0.1, 0.5, 0.9], 
        seed: int = 42,
        n_estimators: int = 200,
        max_depth: int = 15,
        **kwargs
    ):
        super().__init__(quantiles, seed)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.extra_params = kwargs
        self.rf_model = None  # Single model for all quantiles
        
    def fit(self, X:  pd.DataFrame, y: pd.Series):
        """Train Random Forest (extracts quantiles from predictions)."""
        logger.info(f"[RandomForest] Training for quantiles {self.quantiles}")
        
        params = {
            'n_estimators':  self.n_estimators,
            'max_depth': self.max_depth,
            'random_state': self.seed,
            'n_jobs': -1,
            'min_samples_split': 10,
            'min_samples_leaf': 5
        }
        params.update(self.extra_params)
        
        self.rf_model = RandomForestRegressor(**params)
        self.rf_model.fit(X, y)
        
        logger.info(f"[RandomForest] Training completed")
        return self
    
    def predict(self, X: pd.DataFrame) -> Dict[float, np.ndarray]:
        """Predict quantiles using tree predictions."""
        # Suppress sklearn warnings about feature names
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Get predictions from all trees (convert DataFrame to array to avoid warning)
            X_array = X.values if isinstance(X, pd.DataFrame) else X
            all_predictions = np.array([tree.predict(X_array) for tree in self.rf_model.estimators_])
        
        # Extract quantiles
        predictions = {}
        for q in self.quantiles:
            predictions[q] = np.quantile(all_predictions, q, axis=0)
            
        return predictions