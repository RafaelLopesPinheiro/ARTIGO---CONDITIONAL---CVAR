"""
Product-Specific Conformal Prediction.
Calibrates separate conformal corrections for each product to handle heterogeneity.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductSpecificConformalPredictor:
    """
    Product-specific conformal prediction.
    Maintains separate calibration for each product to ensure valid coverage.
    """
    
    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        self.forecaster = None
        self.product_corrections = {}  # Correction per product
        self.products = None
        self.alpha = None
        
    def calibrate(
        self,
        forecaster,
        X_cal: pd.DataFrame,
        y_cal: pd.Series,
        products_cal: pd.Series,
        alpha: float = 0.1
    ):
        """
        Calibrate product-specific conformal corrections.
        
        Args:
            forecaster: Trained quantile forecaster
            X_cal:  Calibration features
            y_cal: Calibration targets
            products_cal: Product labels for calibration set
            alpha: Significance level (1-alpha = coverage)
        """
        logger.info(f"Calibrating Product-Specific Conformal Predictor with alpha={alpha}")
        
        self.forecaster = forecaster
        self.alpha = alpha
        self.products = sorted(products_cal.unique())
        
        logger.info(f"Number of products: {len(self.products)}")
        
        # Reset indices for alignment
        X_cal_reset = X_cal.reset_index(drop=True)
        y_cal_reset = y_cal.reset_index(drop=True)
        products_cal_reset = products_cal.reset_index(drop=True)
        
        # Get base quantile predictions
        cal_preds = forecaster.predict(X_cal_reset)
        
        # Calibrate separately for each product
        logger.info("\nProduct-Specific Calibration:")
        logger.info(f"{'Product':<35} {'n_cal':>8} {'Correction':>12} {'Base_Cov':>10}")
        logger.info("-"*70)
        
        for product in self.products:
            # Filter calibration data for this product
            mask = products_cal_reset == product
            
            if mask.sum() == 0:
                logger.warning(f"No calibration data for {product}, skipping...")
                continue
            
            y_prod = y_cal_reset[mask].values
            pred_lower = cal_preds[0.1][mask]
            pred_upper = cal_preds[0.9][mask]
            
            # Calculate nonconformity scores for this product
            scores = np.maximum(
                pred_lower - y_prod,
                y_prod - pred_upper
            )
            
            # Base coverage (before conformal correction)
            base_coverage = ((y_prod >= pred_lower) & (y_prod <= pred_upper)).mean() * 100
            
            # Compute conformal correction
            n = len(scores)
            q_level = np.ceil((n + 1) * (1 - alpha)) / n
            q_level = min(q_level, 1.0)
            
            correction = np.quantile(scores, q_level)
            
            self.product_corrections[product] = correction
            
            logger.info(f"{product:<35} {n:>8} {correction:>12.3f} {base_coverage:>9.1f}%")
        
        logger.info("\n[OK] Product-specific calibration complete")
        
        return self
    
    def predict(
        self,
        X:  pd.DataFrame,
        products: pd.Series
    ) -> pd.DataFrame:
        """
        Generate predictions with product-specific conformal intervals.
        
        Args:
            X: Features
            products: Product labels
            
        Returns:
            DataFrame with [lower, point, upper]
        """
        # Get base quantile predictions
        preds = self.forecaster.predict(X)
        
        # Reset indices
        X_reset = X.reset_index(drop=True)
        products_reset = products.reset_index(drop=True)
        
        lower = []
        point = []
        upper = []
        
        for idx in range(len(X_reset)):
            product = products_reset.iloc[idx]
            
            # Get product-specific correction
            if product in self.product_corrections:
                correction = self.product_corrections[product]
            else: 
                # Fallback:  use average correction
                correction = np.mean(list(self.product_corrections.values()))
                logger.warning(f"Product {product} not in calibration set, using average correction")
            
            # Apply correction
            lower.append(max(0, preds[0.1][idx] - correction))
            point.append(preds[0.5][idx])
            upper.append(preds[0.9][idx] + correction)
        
        result = pd.DataFrame({
            'lower': lower,
            'point': point,
            'upper':  upper
        })
        
        return result
    
    def get_product_corrections(self) -> pd.DataFrame:
        """Return product-specific corrections as DataFrame."""
        return pd.DataFrame([
            {'product': product, 'correction': correction}
            for product, correction in self.product_corrections.items()
        ]).sort_values('product')


class ProductSpecificCopulaPredictor:
    """
    Combines product-specific calibration with copula modeling.
    """
    
    def __init__(self, seed:  int = 42):
        """
        Args:
            seed:  Random seed
        """
        self.seed = seed
        self.forecaster = None
        self.product_corrections = {}
        self.correlation_matrix = None
        self.products = None
        self.alpha = None
        
    def calibrate(
        self,
        forecaster,
        X_cal: pd.DataFrame,
        y_cal: pd.Series,
        products_cal: pd.Series,
        alpha: float = 0.1
    ):
        """
        Calibrate with both product-specific corrections and copula structure.
        """
        from scipy.stats import norm
        
        logger.info(f"Calibrating Product-Specific Copula Predictor with alpha={alpha}")
        
        self.forecaster = forecaster
        self.alpha = alpha
        self.products = sorted(products_cal.unique())
        
        # Reset indices
        X_cal_reset = X_cal.reset_index(drop=True)
        y_cal_reset = y_cal.reset_index(drop=True)
        products_cal_reset = products_cal.reset_index(drop=True)
        
        # Get base predictions
        cal_preds = forecaster.predict(X_cal_reset)
        
        # Step 1: Product-specific corrections
        nonconformity_scores = {}
        
        logger.info(f"\nCalibrating {len(self.products)} products...")
        
        for product in self.products:
            mask = products_cal_reset == product
            
            if mask.sum() == 0:
                continue
            
            y_prod = y_cal_reset[mask].values
            pred_lower = cal_preds[0.1][mask]
            pred_upper = cal_preds[0.9][mask]
            
            scores = np.maximum(pred_lower - y_prod, y_prod - pred_upper)
            
            n = len(scores)
            q_level = np.ceil((n + 1) * (1 - alpha)) / n
            q_level = min(q_level, 1.0)
            
            correction = np.quantile(scores, q_level)
            self.product_corrections[product] = correction
            nonconformity_scores[product] = scores
        
        # Step 2: Estimate copula correlation from nonconformity scores
        # Align scores across products
        n_samples = min(len(scores) for scores in nonconformity_scores.values())
        
        score_matrix = np.column_stack([
            nonconformity_scores[product][: n_samples]
            for product in self.products
            if product in nonconformity_scores
        ])
        
        # Transform to uniform margins
        uniform_scores = np.zeros_like(score_matrix)
        for i in range(score_matrix.shape[1]):
            sorted_scores = np.sort(score_matrix[:, i])
            ranks = np.searchsorted(sorted_scores, score_matrix[:, i], side='right')
            uniform_scores[:, i] = ranks / (len(sorted_scores) + 1)
        
        # Transform to Gaussian margins
        gaussian_scores = norm.ppf(np.clip(uniform_scores, 0.001, 0.999))
        
        # Estimate correlation
        self.correlation_matrix = np.corrcoef(gaussian_scores.T)
        
        logger.info(f"\nEstimated Copula Correlation Matrix (shape: {self.correlation_matrix.shape})")
        logger.info(f"Mean absolute correlation: {np.abs(self.correlation_matrix[np.triu_indices_from(self.correlation_matrix, k=1)]).mean():.3f}")
        
        logger.info("\n[OK] Product-specific copula calibration complete")
        
        return self
    
    def predict(
        self,
        X: pd.DataFrame,
        products: pd.Series
    ) -> pd.DataFrame:
        """
        Generate predictions with product-specific copula-based intervals.
        """
        # Get base predictions
        preds = self.forecaster.predict(X)
        
        # Reset indices
        products_reset = products.reset_index(drop=True)
        
        lower = []
        point = []
        upper = []
        
        for idx in range(len(X)):
            product = products_reset.iloc[idx]
            
            if product in self.product_corrections:
                correction = self.product_corrections[product]
            else: 
                correction = np.mean(list(self.product_corrections.values()))
            
            lower.append(max(0, preds[0.1][idx] - correction))
            point.append(preds[0.5][idx])
            upper.append(preds[0.9][idx] + correction)
        
        result = pd.DataFrame({
            'lower': lower,
            'point': point,
            'upper': upper
        })
        
        return result
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """Return correlation matrix as DataFrame."""
        return pd.DataFrame(
            self.correlation_matrix,
            index=self.products,
            columns=self.products
        )