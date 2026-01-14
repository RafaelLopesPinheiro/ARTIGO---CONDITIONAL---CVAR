"""
CatBoost implementation for quantile forecasting.
"""

from catboost import CatBoostRegressor, Pool
import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from .base_model import BaseQuantileModel

logger = logging.getLogger(__name__)


class CatBoostQuantileModel(BaseQuantileModel):
    """CatBoost-based quantile forecasting."""
    
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
        
    def fit(self, X:  pd.DataFrame, y: pd.Series):
        """Train CatBoost models for each quantile."""
        logger.info(f"[CatBoost] Training for quantiles {self.quantiles}")
        
        for q in self.quantiles:
            params = {
                'loss_function': 'Quantile',  # NOT 'Quantile: alpha=X'
                'iterations': self.n_estimators,
                'depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'random_seed': self.seed,
                'verbose': False,
                'allow_writing_files': False
            }
            
            # FIXED: Pass alpha as separate parameter
            model = CatBoostRegressor(**params)
            model.set_params(loss_function=f'Quantile: alpha={q}')  # This way works
            model.fit(X, y)
            self.models[q] = model
            
        logger.info(f"[CatBoost] Training completed")
        return self
    
    def predict(self, X: pd.DataFrame) -> Dict[float, np.ndarray]:
        """Predict quantiles."""
        predictions = {}
        for q in self.quantiles:
            predictions[q] = self.models[q].predict(X)
        return predictions