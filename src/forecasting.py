"""
Módulo de previsão probabilística usando Conformal Prediction.
Implementa quantile forecasting com LightGBM e intervalos conformais.
UPDATED: Now supports multiple model types via BaseQuantileModel interface.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Union
import lightgbm as lgb
from sklearn.base import BaseEstimator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantileForecaster(BaseEstimator):
    """
    Forecaster baseado em quantis usando LightGBM.
    DEPRECATED: Use models from src.models instead.
    Kept for backward compatibility.
    """
    
    def __init__(self, quantiles: List[float] = [0.1, 0.5, 0.9], seed: int = 42, params: Dict = None):
        """
        Args:
            quantiles: Lista de quantis para prever
            seed: Seed para reprodutibilidade
            params: Dicionário com parâmetros do LightGBM (opcional)
        """
        self.quantiles = quantiles
        self.seed = seed
        self.params = params or {}
        self.models = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Treina modelos de quantis.
        
        Args:
            X: Features
            y: Target
        """
        logger.info(f"Treinando modelos para quantis {self.quantiles}")
        
        # Parâmetros base
        base_params = {
            'objective': 'quantile',
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate':  0.05,
            'num_leaves': 31,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.seed,
            'verbose': -1
        }
        
        # Mesclar com parâmetros customizados
        base_params.update(self.params)
        
        for q in self.quantiles:
            logger.info(f"Treinando modelo para quantil {q}")
            params = base_params.copy()
            params['alpha'] = q
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X, y)
            
            self.models[q] = model
            
        logger.info("Treinamento concluído")
        return self
    
    def predict(self, X: pd.DataFrame) -> Dict[float, np.ndarray]:
        """
        Prevê quantis para novos dados.
        
        Args:
            X: Features
            
        Returns:
            Dicionário {quantil: predições}
        """
        predictions = {}
        
        for q in self.quantiles:
            predictions[q] = self.models[q].predict(X)
            
        return predictions


class ConformalPredictor:
    """
    Implementa Split Conformal Prediction para intervalos de previsão.
    UPDATED: Now works with any model that implements predict() returning Dict[float, np.ndarray]
    """
    
    def __init__(self, seed: int = 42):
        """
        Args:
            seed:  Seed para reprodutibilidade
        """
        self.seed = seed
        self.forecaster = None
        self.correction = None
        self.quantiles = None
        self.alpha = None
        
    def calibrate(
        self,
        forecaster: Union[QuantileForecaster, BaseEstimator],  # Accept any model
        X_cal: pd.DataFrame,
        y_cal: pd.Series,
        alpha: float = 0.1
    ):
        """
        Calibra intervalos usando conjunto de calibração.
        
        Args:
            forecaster: Modelo já treinado (must have predict() method)
            X_cal: Features de calibração
            y_cal:  Targets de calibração
            alpha: Nível de significância (1-alpha = cobertura desejada)
        """
        logger.info(f"Calibrando intervalos com alpha={alpha}")
        
        self.forecaster = forecaster
        self.alpha = alpha
        self.quantiles = forecaster.quantiles
        
        # Obter predições de calibração
        pred_dict = forecaster.predict(X_cal)
        
        # Encontrar quantil inferior e superior
        q_lower = min(self.quantiles)
        q_upper = max(self.quantiles)
        
        y_pred_lower = pred_dict[q_lower]
        y_pred_upper = pred_dict[q_upper]
        
        # Calcular não-conformidade (non-conformity scores)
        # Score = max(lower - y, y - upper)
        scores = np.maximum(y_pred_lower - y_cal.values, y_cal.values - y_pred_upper)
        
        # Calcular quantil de correção conformal
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        self.correction = np.quantile(scores, q_level)
        
        logger.info(f"Correcao conformal: {self.correction:.2f}")
        logger.info(f"Quantile level usado: {q_level:.4f}")
        
        return self
    
    def predict(self, X:  pd.DataFrame) -> pd.DataFrame:
        """
        Generate conformal prediction intervals.
        
        Returns:
            DataFrame with columns ['lower', 'point', 'upper'] and len(X) rows
        """
        # Get base quantile predictions (returns dict with keys 0.1, 0.5, 0.9)
        base_preds = self.forecaster.predict(X)
        
        # ✅ FIX:  Ensure base_preds are arrays, not DataFrames
        if isinstance(base_preds, dict):
            # Extract values and ensure they're 1D arrays
            lower_base = np.array(base_preds[0.1]).flatten()
            point_base = np.array(base_preds[0.5]).flatten()
            upper_base = np.array(base_preds[0.9]).flatten()
        else:
            raise ValueError(f"Expected dict from forecaster.predict(), got {type(base_preds)}")
        
        # Apply conformal correction
        lower = np.maximum(0, lower_base - self.correction)
        point = point_base
        upper = upper_base + self.correction
        
        # Create DataFrame with correct shape (n_samples, 3)
        result = pd.DataFrame({
            'lower': lower,
            'point':  point,
            'upper': upper
        })
        
        # ✅ VALIDATE: Check shape
        if len(result) != len(X):
            raise ValueError(f"Prediction shape mismatch: got {len(result)} rows, expected {len(X)}")
        
        return result


def evaluate_forecast(y_true: pd.Series, predictions: pd.DataFrame) -> dict:
    """
    Evaluate forecast quality. 
    
    Args:
        y_true:  Actual values
        predictions: DataFrame with columns ['lower', 'point', 'upper']
    
    Returns: 
        Dict with metrics
    """
    # ✅ FIX: Use 'point' instead of 'median'
    y_pred = predictions['point']. values
    y_true_values = y_true.values
    
    # Calculate metrics
    mae = np.mean(np.abs(y_true_values - y_pred))
    rmse = np.sqrt(np.mean((y_true_values - y_pred)**2))
    
    # MAPE (avoid division by zero)
    mape = np.mean(np.abs((y_true_values - y_pred) / np.maximum(y_true_values, 1e-10))) * 100
    
    # Coverage
    covered = (y_true_values >= predictions['lower'].values) & (y_true_values <= predictions['upper'].values)
    coverage = covered.mean() * 100
    
    # Average interval width
    avg_width = (predictions['upper'].values - predictions['lower'].values).mean()
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'Coverage (%)': coverage,
        'Avg Interval Width': avg_width
    }

def validate_coverage(self, X_val, y_val):
    """
    Validate coverage on a held-out validation set.
    Helps detect distribution shift.
    """
    predictions = self.predict(X_val)
    covered = ((y_val >= predictions['lower']) & 
               (y_val <= predictions['upper']))
    actual_coverage = covered.mean()
    target_coverage = 1 - self.alpha
    
    gap = actual_coverage - target_coverage
    
    logger.info(f"\n📊 Coverage Validation:")
    logger.info(f"  Target: {target_coverage*100:.1f}%")
    logger.info(f"  Actual: {actual_coverage*100:.1f}%")
    logger.info(f"  Gap: {gap*100:+.1f}pp")
    
    if abs(gap) > 0.05:  # More than 5pp difference
        logger.warning(f"⚠️  Coverage gap > 5pp detected!")
        logger.warning(f"   Possible distribution shift between calibration and test")
    
    return actual_coverage