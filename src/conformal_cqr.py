"""
Conformalized Quantile Regression (CQR) for robust conformal prediction.
More robust to distribution shift and heteroscedastic errors than split conformal.

Reference: Romano, Patterson, and Candes (2019)
"Conformalized Quantile Regression" (NeurIPS 2019)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CQRPredictor:
    """
    Conformalized Quantile Regression (CQR).
    
    Key Innovation:Asymmetric conformalization
    - Separate calibration for lower and upper bounds
    - More robust to distribution shift
    - Handles heteroscedastic errors better
    """
    
    def __init__(self, seed:int = 42):
        """
        Args:
            seed:Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        self.forecaster = None
        self.correction_low = None
        self.correction_high = None
        self.alpha = None
        
    def calibrate(
        self,
        forecaster,
        X_cal:pd.DataFrame,
        y_cal:pd.Series,
        alpha:float = 0.1
    ):
        """
        Calibrate CQR using asymmetric nonconformity scores.
        
        Args:
            forecaster: Trained quantile forecaster
            X_cal:Calibration features
            y_cal:Calibration targets
            alpha:Significance level (1-alpha = coverage)
        """
        logger.info(f"Calibrating CQR with alpha={alpha}")
        
        self.forecaster = forecaster
        self.alpha = alpha
        
        # Get quantile predictions on calibration set
        cal_preds = forecaster.predict(X_cal)
        
        y_cal_values = y_cal.values
        q_low = cal_preds[0.1]  # Lower quantile (10%)
        q_high = cal_preds[0.9]  # Upper quantile (90%)
        
        # ====================================================================
        # CQR KEY INNOVATION:Asymmetric nonconformity scores
        # ====================================================================
        
        # Lower bound errors: how much do we under-predict the lower tail?
        scores_low = q_low - y_cal_values
        
        # Upper bound errors:how much do we under-predict the upper tail?
        scores_high = y_cal_values - q_high
        
        logger.info(f"\nNonconformity Scores (Asymmetric):")
        logger.info(f"  Lower bound scores:")
        logger.info(f"    Mean:{scores_low.mean():.3f}")
        logger.info(f"    Std:{scores_low.std():.3f}")
        logger.info(f"    Quantile range:[{np.quantile(scores_low, 0.1):.3f}, {np.quantile(scores_low, 0.9):.3f}]")
        
        logger.info(f"  Upper bound scores:")
        logger.info(f"    Mean:{scores_high.mean():.3f}")
        logger.info(f"    Std:{scores_high.std():.3f}")
        logger.info(f"    Quantile range:[{np.quantile(scores_high, 0.1):.3f}, {np.quantile(scores_high, 0.9):.3f}]")
        
        # Compute separate corrections for lower and upper bounds
        n = len(scores_low)
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        q_level = min(q_level, 1.0)
        
        self.correction_low = np.quantile(scores_low, q_level)
        self.correction_high = np.quantile(scores_high, q_level)
        
        logger.info(f"\nCQR Corrections:")
        logger.info(f"  Lower bound correction:{self.correction_low:.3f}")
        logger.info(f"  Upper bound correction: {self.correction_high:.3f}")
        logger.info(f"  Asymmetry ratio:{self.correction_high / max(abs(self.correction_low), 1e-6):.2f}")
        
        # Base coverage (before CQR)
        base_coverage = ((y_cal_values >= q_low) & (y_cal_values <= q_high)).mean() * 100
        logger.info(f"\nBase quantile coverage:{base_coverage:.2f}%")
        logger.info(f"Target coverage:{(1 - alpha) * 100:.2f}%")
        
        logger.info("\n[OK] CQR calibration complete")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate CQR prediction intervals.
        
        Args:
            X:Features
            
        Returns:
            DataFrame with [lower, point, upper]
        """
        # Get base quantile predictions
        preds = self.forecaster.predict(X)
        
        # Apply asymmetric CQR corrections
        lower = np.maximum(0, preds[0.1] - self.correction_low)
        point = preds[0.5]
        upper = preds[0.9] + self.correction_high
        
        result = pd.DataFrame({
            'lower':lower,
            'point':point,
            'upper': upper
        })
        
        return result


class ProductSpecificCQR:
    """
    Product-Specific Conformalized Quantile Regression.
    Combines CQR with product-specific calibration.
    """
    
    def __init__(self, seed:int = 42):
        """
        Args:
            seed:Random seed
        """
        self.seed = seed
        np.random.seed(seed)
        self.forecaster = None
        self.product_corrections = {}  # {product: (correction_low, correction_high)}
        self.products = None
        self.alpha = None
        
    def calibrate(
        self,
        forecaster,
        X_cal:pd.DataFrame,
        y_cal:pd.Series,
        products_cal:pd.Series,
        alpha:float = 0.1
    ):
        """
        Calibrate product-specific CQR.
        
        Args:
            forecaster: Trained quantile forecaster
            X_cal:Calibration features
            y_cal:Calibration targets
            products_cal: Product labels
            alpha:Significance level
        """
        logger.info(f"Calibrating Product-Specific CQR with alpha={alpha}")
        
        self.forecaster = forecaster
        self.alpha = alpha
        self.products = sorted(products_cal.unique())
        
        logger.info(f"Number of products:{len(self.products)}")
        
        # Reset indices
        X_cal_reset = X_cal.reset_index(drop=True)
        y_cal_reset = y_cal.reset_index(drop=True)
        products_cal_reset = products_cal.reset_index(drop=True)
        
        # Get base predictions
        cal_preds = forecaster.predict(X_cal_reset)
        
        # Calibrate separately for each product
        logger.info("\nProduct-Specific CQR Calibration:")
        logger.info(f"{'Product':<35} {'n_cal':>8} {'Corr_Low':>10} {'Corr_High':>11} {'Asym':>6}")
        logger.info("-"*80)
        
        for product in self.products:
            # Filter for this product
            mask = products_cal_reset == product
            
            if mask.sum() == 0:
                logger.warning(f"No calibration data for {product}")
                continue
            
            y_prod = y_cal_reset[mask].values
            q_low = cal_preds[0.1][mask]
            q_high = cal_preds[0.9][mask]
            
            # Asymmetric scores
            scores_low = q_low - y_prod
            scores_high = y_prod - q_high
            
            # Compute corrections
            n = len(scores_low)
            q_level = np.ceil((n + 1) * (1 - alpha)) / n
            q_level = min(q_level, 1.0)
            
            correction_low = np.quantile(scores_low, q_level)
            correction_high = np.quantile(scores_high, q_level)
            
            self.product_corrections[product] = (correction_low, correction_high)
            
            # Asymmetry measure
            asymmetry = correction_high / max(abs(correction_low), 1e-6)
            
            logger.info(f"{product:<35} {n:>8} {correction_low:>10.3f} {correction_high:>11.3f} {asymmetry:>6.2f}")
        
        logger.info("\n[OK] Product-specific CQR calibration complete")
        
        return self
    
    def predict(
        self,
        X:pd.DataFrame,
        products:pd.Series
    ) -> pd.DataFrame:
        """
        Generate product-specific CQR predictions.
        
        Args:
            X:Features
            products:Product labels
            
        Returns:
            DataFrame with [lower, point, upper]
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
            
            # Get product-specific corrections
            if product in self.product_corrections:
                correction_low, correction_high = self.product_corrections[product]
            else:
                # Fallback: average corrections
                all_corr_low = [c[0] for c in self.product_corrections.values()]
                all_corr_high = [c[1] for c in self.product_corrections.values()]
                correction_low = np.mean(all_corr_low)
                correction_high = np.mean(all_corr_high)
            
            # Apply asymmetric corrections
            lower.append(max(0, preds[0.1][idx] - correction_low))
            point.append(preds[0.5][idx])
            upper.append(preds[0.9][idx] + correction_high)
        
        result = pd.DataFrame({
            'lower':lower,
            'point': point,
            'upper':upper
        })
        
        return result
    
    def get_corrections_summary(self) -> pd.DataFrame:
        """Return summary of product-specific corrections."""
        summary = []
        for product, (corr_low, corr_high) in self.product_corrections.items():
            summary.append({
                'product':product,
                'correction_low':corr_low,
                'correction_high':corr_high,
                'asymmetry_ratio':corr_high / max(abs(corr_low), 1e-6),
                'total_width_adjustment':abs(corr_low) + corr_high
            })
        
        return pd.DataFrame(summary).sort_values('product')