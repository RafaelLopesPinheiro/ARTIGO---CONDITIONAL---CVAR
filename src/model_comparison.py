"""
Module for comparing multiple forecasting models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Type
import logging
import time
from pathlib import Path

from src.models.base_model import BaseQuantileModel
from src.models import *
from src.forecasting import ConformalPredictor, evaluate_forecast

logger = logging.getLogger(__name__)


class ModelComparison:
    """Compare multiple quantile forecasting models."""
    
    def __init__(self, models_config: Dict[str, Dict], quantiles: List[float], seed: int = 42):
        """
        Args:
            models_config: Dict with {model_name: {class:  ModelClass, params: {...}}}
            quantiles: List of quantiles to forecast
            seed: Random seed
        """
        self.models_config = models_config
        self.quantiles = quantiles
        self.seed = seed
        self.results = {}
        self.trained_models = {}
        
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train all configured models."""
        logger.info("="*80)
        logger.info("TRAINING ALL MODELS")
        logger.info("="*80)
        
        for model_name, config in self.models_config.items():
            logger.info(f"\n[{model_name}] Starting training...")
            start_time = time.time()
            
            try:
                # Instantiate model
                model_class = config['class']
                model_params = config.get('params', {})
                model = model_class(quantiles=self.quantiles, seed=self.seed, **model_params)
                
                # Train
                model.fit(X_train, y_train)
                
                training_time = time.time() - start_time
                self.trained_models[model_name] = model
                
                logger.info(f"[{model_name}] ✅ Training completed in {training_time:.2f}s")
                
            except Exception as e:
                logger.error(f"[{model_name}] ❌ Training failed: {str(e)}")
                self.trained_models[model_name] = None
                
        logger.info(f"\n✅ Trained {len([m for m in self.trained_models.values() if m is not None])}/{len(self.models_config)} models successfully")
        
    def calibrate_all_models(
        self, 
        X_cal: pd.DataFrame, 
        y_cal: pd.Series, 
        alpha: float = 0.1
    ):
        """Calibrate conformal predictors for all models."""
        logger.info("\n" + "="*80)
        logger.info("CALIBRATING CONFORMAL PREDICTORS")
        logger.info("="*80)
        
        self.conformal_predictors = {}
        
        for model_name, model in self.trained_models.items():
            if model is None: 
                continue
                
            logger.info(f"\n[{model_name}] Calibrating...")
            
            try:
                cp = ConformalPredictor(seed=self.seed)
                cp.calibrate(model, X_cal, y_cal, alpha=alpha)
                self.conformal_predictors[model_name] = cp
                logger.info(f"[{model_name}] ✅ Calibration completed (correction={cp.correction:.2f})")
                
            except Exception as e:
                logger.error(f"[{model_name}] ❌ Calibration failed: {str(e)}")
                
    def evaluate_all_models(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        df_test: pd.DataFrame
    ):
        """Evaluate all models on test set."""
        logger.info("\n" + "="*80)
        logger.info("EVALUATING ALL MODELS")
        logger.info("="*80)
        
        results_list = []
        
        for model_name in self.trained_models.keys():
            if model_name not in self.conformal_predictors:
                continue
                
            logger.info(f"\n[{model_name}] Evaluating...")
            
            try:
                cp = self.conformal_predictors[model_name]
                
                # Predict
                start_time = time.time()
                pred_intervals = cp.predict(X_test)
                prediction_time = time.time() - start_time
                
                # Evaluate
                metrics = evaluate_forecast(
                    y_true=y_test,
                    y_pred_lower=pred_intervals['lower'],
                    y_pred_median=pred_intervals['median'],
                    y_pred_upper=pred_intervals['upper']
                )
                
                # Store results
                result = {
                    'model':  model_name,
                    'mae': metrics['mae'],
                    'rmse': metrics['rmse'],
                    'mape': metrics['mape'],
                    'coverage': metrics['coverage'],
                    'avg_interval_width': metrics['avg_interval_width'],
                    'prediction_time': prediction_time,
                    **pred_intervals
                }
                
                results_list.append(result)
                self.results[model_name] = result
                
                logger.info(f"[{model_name}] Metrics:")
                logger.info(f"  MAE: {metrics['mae']:.2f}")
                logger.info(f"  RMSE: {metrics['rmse']:.2f}")
                logger.info(f"  MAPE: {metrics['mape']:.2f}%")
                logger.info(f"  Coverage: {metrics['coverage']:.2f}%")
                logger.info(f"  Avg Interval Width: {metrics['avg_interval_width']:.2f}")
                logger.info(f"  Prediction Time: {prediction_time:.3f}s")
                
            except Exception as e:
                logger.error(f"[{model_name}] ❌ Evaluation failed: {str(e)}")
                
        # Create comparison DataFrame
        self.comparison_df = pd.DataFrame(results_list)
        
        if len(self.comparison_df) > 0:
            # Rank models
            self.comparison_df['rank_mae'] = self.comparison_df['mae'].rank()
            self.comparison_df['rank_rmse'] = self.comparison_df['rmse'].rank()
            self.comparison_df['rank_coverage'] = (100 - abs(self.comparison_df['coverage'] - 95)).rank(ascending=False)
            self.comparison_df['rank_width'] = self.comparison_df['avg_interval_width'].rank()
            
            # Overall rank (lower is better)
            self.comparison_df['overall_rank'] = (
                self.comparison_df['rank_mae'] + 
                self.comparison_df['rank_rmse'] + 
                self.comparison_df['rank_coverage'] +
                self.comparison_df['rank_width']
            ) / 4
            
            self.comparison_df = self.comparison_df.sort_values('overall_rank')
            
        return self.comparison_df
    
    def get_best_model(self, metric: str = 'overall_rank') -> str:
        """Return name of best performing model."""
        if len(self.comparison_df) == 0:
            return None
            
        if metric == 'overall_rank':
            return self.comparison_df.iloc[0]['model']
        else:
            ascending = metric not in ['coverage']
            return self.comparison_df.sort_values(metric, ascending=ascending).iloc[0]['model']
    
    def print_comparison_table(self):
        """Print formatted comparison table."""
        logger.info("\n" + "="*80)
        logger.info("MODEL COMPARISON RESULTS")
        logger.info("="*80)
        
        if len(self.comparison_df) == 0:
            logger.warning("No results to display")
            return
            
        # Select columns for display
        display_cols = ['model', 'mae', 'rmse', 'mape', 'coverage', 'avg_interval_width', 'overall_rank']
        display_df = self.comparison_df[display_cols].copy()
        
        # Format
        display_df['mae'] = display_df['mae'].apply(lambda x: f"{x:.2f}")
        display_df['rmse'] = display_df['rmse'].apply(lambda x: f"{x:.2f}")
        display_df['mape'] = display_df['mape'].apply(lambda x: f"{x:.2f}%")
        display_df['coverage'] = display_df['coverage'].apply(lambda x: f"{x:.2f}%")
        display_df['avg_interval_width'] = display_df['avg_interval_width'].apply(lambda x: f"{x:.2f}")
        display_df['overall_rank'] = display_df['overall_rank'].apply(lambda x: f"{x:.2f}")
        
        logger.info(f"\n{display_df.to_string(index=False)}")
        
        best_model = self.get_best_model()
        logger.info(f"\n🏆 BEST MODEL: {best_model}")
        
    def save_results(self, output_dir: Path):
        """Save comparison results."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comparison table
        self.comparison_df.to_csv(output_dir / 'model_comparison.csv', index=False)
        logger.info(f"Results saved to {output_dir / 'model_comparison.csv'}")