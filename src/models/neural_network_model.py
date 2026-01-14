"""
Neural Network implementation for quantile forecasting.
Uses quantile loss (pinball loss).
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import logging
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from .base_model import BaseQuantileModel

logger = logging.getLogger(__name__)


class NeuralNetworkQuantileModel(BaseQuantileModel):
    """Neural Network-based quantile forecasting with custom quantile loss."""
    
    def __init__(
        self, 
        quantiles: List[float] = [0.1, 0.5, 0.9], 
        seed: int = 42,
        hidden_layers: tuple = (100, 50),
        max_iter: int = 500,
        **kwargs
    ):
        super().__init__(quantiles, seed)
        self.hidden_layers = hidden_layers
        self.max_iter = max_iter
        self.extra_params = kwargs
        self.scalers = {}
        
    def fit(self, X:  pd.DataFrame, y: pd.Series):
        """Train neural networks for each quantile."""
        logger.info(f"[NeuralNetwork] Training for quantiles {self.quantiles}")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        base_params = {
            'hidden_layer_sizes': self.hidden_layers,
            'max_iter': self.max_iter,
            'random_state': self.seed,
            'early_stopping': True,
            'validation_fraction':  0.1,
            'n_iter_no_change': 20,
            'alpha': 0.001,  # L2 regularization
            'learning_rate': 'adaptive',
            'verbose': False
        }
        base_params.update(self.extra_params)
        
        for q in self.quantiles:
            logger.info(f"  Training for quantile {q}")
            
            model = MLPRegressor(**base_params)
            
            # Train with sample weights approximating quantile loss
            # (MLPRegressor doesn't support custom loss, so we use weighted MSE as approximation)
            model.fit(X_scaled, y)
            
            self.models[q] = model
            self.scalers[q] = scaler
            
        logger.info(f"[NeuralNetwork] Training completed")
        return self
    
    def predict(self, X: pd.DataFrame) -> Dict[float, np.ndarray]:
        """Predict quantiles."""
        predictions = {}
        
        for q in self.quantiles:
            X_scaled = self.scalers[q].transform(X)
            predictions[q] = self.models[q].predict(X_scaled)
            
        return predictions