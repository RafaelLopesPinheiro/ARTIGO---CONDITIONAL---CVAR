"""
Módulo de previsão probabilística usando Conformal Prediction.
Implementa quantile forecasting com LightGBM e intervalos conformais.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import lightgbm as lgb
from sklearn.base import BaseEstimator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantileForecaster(BaseEstimator):
    """
    Forecaster baseado em quantis usando LightGBM.
    """
    
    def __init__(self, quantiles:  List[float] = [0.1, 0.5, 0.9], seed: int = 42, params: Dict = None):
        """
        Args:
            quantiles: Lista de quantis para prever
            seed: Seed para reprodutibilidade
            params:  Dicionário com parâmetros do LightGBM (opcional)
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
            'min_child_samples':  20,
            'subsample': 0.8,
            'colsample_bytree':  0.8,
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
    """
    
    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Seed para reprodutibilidade
        """
        self.seed = seed
        self.forecaster = None
        self.correction = None
        
    def calibrate(
        self,
        forecaster:  QuantileForecaster,
        X_cal: pd.DataFrame,
        y_cal: pd.Series,
        alpha: float = 0.1
    ):
        """
        Calibra intervalos usando conjunto de calibração.
        
        Args:
            forecaster: Modelo já treinado
            X_cal:  Features de calibração
            y_cal: Targets de calibração
            alpha:  Nível de significância (1-alpha = coverage)
        """
        logger.info(f"Calibrando intervalos com alpha={alpha}")
        
        self.forecaster = forecaster
        self.alpha = alpha
        
        # Prever quantis no conjunto de calibração
        cal_preds = forecaster.predict(X_cal)
        
        # Calcular nonconformity scores (max deviation)
        scores = np.maximum(
            cal_preds[0.1] - y_cal.values,
            y_cal.values - cal_preds[0.9]
        )
        
        # FIX: Usar fórmula ajustada para melhor coverage
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        q_level = min(q_level, 1.0)
        
        self.correction = np.quantile(scores, q_level)
        
        logger.info(f"Correcao conformal: {self.correction:.2f}")
        logger.info(f"Quantile level usado: {q_level:.4f}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Prevê com intervalos conformais.
        
        Args:
            X: Features
            
        Returns:
            DataFrame com colunas [lower, point, upper]
        """
        # Prever quantis base
        preds = self.forecaster.predict(X)
        
        # Aplicar correção conformal
        lower = np.maximum(0, preds[0.1] - self.correction)
        point = preds[0.5]
        upper = preds[0.9] + self.correction
        
        result = pd.DataFrame({
            'lower': lower,
            'point': point,
            'upper':  upper
        })
        
        return result


def evaluate_forecast(y_true: pd.Series, predictions: pd.DataFrame) -> Dict[str, float]:
    """
    Avalia qualidade das previsões.
    
    Args:
        y_true:  Valores reais
        predictions: DataFrame com [lower, point, upper]
        
    Returns:
        Dicionário com métricas
    """
    logger.info("Avaliando previsões")
    
    y_true = y_true.values
    point_pred = predictions['point'].values
    lower = predictions['lower'].values
    upper = predictions['upper'].values
    
    # Métricas pontuais
    mae = np.mean(np.abs(y_true - point_pred))
    rmse = np.sqrt(np.mean((y_true - point_pred) ** 2))
    mape = np.mean(np.abs((y_true - point_pred) / (y_true + 1))) * 100  # +1 para evitar divisão por zero
    
    # Métricas de intervalo
    coverage = np.mean((y_true >= lower) & (y_true <= upper)) * 100
    avg_width = np.mean(upper - lower)
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'Coverage (%)': coverage,
        'Avg Interval Width': avg_width
    }
    
    logger.info("Métricas de previsão:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.2f}")
    
    return metrics