"""
Base class for quantile forecasting models.
All models must inherit from this to ensure consistent interface.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.base import BaseEstimator


class BaseQuantileModel(ABC, BaseEstimator):
    """Abstract base class for quantile forecasting models."""
    
    def __init__(self, quantiles:  List[float] = [0.1, 0.5, 0.9], seed: int = 42):
        """
        Args:
            quantiles: List of quantiles to forecast
            seed: Random seed for reproducibility
        """
        self.quantiles = quantiles
        self.seed = seed
        self.models = {}
        self.model_name = self.__class__.__name__
        
    @abstractmethod
    def fit(self, X:  pd.DataFrame, y: pd.Series):
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Dict[float, np.ndarray]:
        """
        Predict quantiles for new data.
        
        Returns:
            Dictionary {quantile: predictions}
        """
        pass
    
    def get_params(self, deep=True) -> Dict: 
        """Return model parameters (sklearn compatible)."""
        return {
            'quantiles': self.quantiles,
            'seed':  self.seed
        }
    
    def set_params(self, **params):
        """Set model parameters (sklearn compatible)."""
        for key, value in params.items():
            setattr(self, key, value)
        return self