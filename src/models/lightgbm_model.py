"""
LightGBM implementation for quantile forecasting.
"""

import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from .base_model import BaseQuantileModel

logger = logging.getLogger(__name__)


class LightGBMQuantileModel(BaseQuantileModel):
    """LightGBM-based quantile forecasting."""
    
    def __init__(
        self, 
        quantiles: List[float] = [0.1, 0.5, 0.9], 
        seed: int = 42,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        **kwargs
    ):
        super().__init__(quantiles, seed)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.extra_params = kwargs
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train LightGBM models for each quantile."""
        logger.info(f"[LightGBM] Training for quantiles {self.quantiles}")
        
        base_params = {
            'objective': 'quantile',
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'num_leaves': 31,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.seed,
            'verbose': -1
        }
        base_params.update(self.extra_params)
        
        for q in self.quantiles:
            params = base_params.copy()
            params['alpha'] = q
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X, y)
            self.models[q] = model
            
        logger.info(f"[LightGBM] Training completed")
        return self
    
    def predict(self, X: pd.DataFrame) -> Dict[float, np.ndarray]:
        """Predict quantiles."""
        predictions = {}
        for q in self.quantiles:
            predictions[q] = self.models[q].predict(X)
        return predictions