"""
Linear Quantile Regression implementation.
"""

from sklearn.linear_model import QuantileRegressor
import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from .base_model import BaseQuantileModel

logger = logging.getLogger(__name__)


class LinearQuantileModel(BaseQuantileModel):
    """Linear Quantile Regression."""
    
    def __init__(
        self, 
        quantiles: List[float] = [0.1, 0.5, 0.9], 
        seed: int = 42,
        alpha: float = 0.0,  # Regularization
        **kwargs
    ):
        super().__init__(quantiles, seed)
        self.alpha = alpha
        self.extra_params = kwargs
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train linear quantile models."""
        logger.info(f"[LinearQuantile] Training for quantiles {self.quantiles}")
        
        params = {
            'alpha':  self.alpha,
            'solver': 'highs'
        }
        params.update(self.extra_params)
        
        for q in self.quantiles:
            model = QuantileRegressor(quantile=q, **params)
            model.fit(X, y)
            self.models[q] = model
            
        logger.info(f"[LinearQuantile] Training completed")
        return self
    
    def predict(self, X: pd.DataFrame) -> Dict[float, np.ndarray]: 
        """Predict quantiles."""
        predictions = {}
        for q in self.quantiles:
            predictions[q] = self.models[q].predict(X)
        return predictions