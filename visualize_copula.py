"""
Visualize the Gaussian Copula structure estimated from the data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

import config
from src.data_preparation import load_data, create_features, split_data
from src.forecasting import QuantileForecaster
from conformal_product_specific import ProductSpecificConformalPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def visualize_copula_structure(output_dir='outputs/copula_analysis'):
    """Visualize the copula structure and dependencies."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    df = load_data(config.DATA_PATH)
    df_features = create_features(df)
    X_train, y_train, X_cal, y_cal, X_test, y_test = split_data(
        df_features,
        train_end_date=config.TRAIN_END_DATE,
        test_start_date=config.TEST_START_DATE,
        calibration_months=config.CALIBRATION_MONTHS
    )
    
    products_cal = df_features[df_features.index.isin(X_cal.index)]['product']
    
    # Train forecaster
    logger.info("Training forecaster...")
    qf = QuantileForecaster(
        quantiles=config.QUANTILES,
        seed=config.SEED,
        params=config.LGBM_PARAMS
    )
    qf.fit(X_train, y_train)
    
    # Calibrate copula
    logger.info("Calibrating copula...")
    cp_copula = ProductSpecificConformalPredictor(seed=config.SEED)
    cp_copula.calibrate(qf, X_cal, y_cal, products_cal, alpha=config.ALPHA_CONFORMAL)
    
    # Get correlation matrix
    corr_matrix = cp_copula.get_correlation_matrix()
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Heatmap
    ax = axes[0]
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title('Gaussian Copula Correlation Matrix\n(Nonconformity Scores)', 
                 fontsize=13, fontweight='bold')
    
    # Network graph
    ax = axes[1]
    products = corr_matrix.index.tolist()
    n_products = len(products)
    
    # Position products in a circle
    angles = np.linspace(0, 2*np.pi, n_products, endpoint=False)
    x_pos = np.cos(angles)
    y_pos = np.sin(angles)
    
    # Draw edges for strong correlations
    threshold = 0.3
    for i in range(n_products):
        for j in range(i+1, n_products):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > threshold:
                alpha = min(abs(corr), 1.0)
                color = 'red' if corr > 0 else 'blue'
                linewidth = abs(corr) * 3
                ax.plot([x_pos[i], x_pos[j]], [y_pos[i], y_pos[j]], 
                       color=color, alpha=alpha, linewidth=linewidth, zorder=1)
    
    # Draw nodes
    ax.scatter(x_pos, y_pos, s=500, c='lightblue', edgecolors='black', 
              linewidths=2, zorder=2)
    
    # Add labels
    for i, product in enumerate(products):
        # Shorten labels
        label = product[: 15] + '...' if len(product) > 15 else product
        # Position labels outside circle
        offset = 1.2
        ax.text(x_pos[i]*offset, y_pos[i]*offset, label, 
               ha='center', va='center', fontsize=8, fontweight='bold')
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Dependency Network (|correlation| > {threshold})\nRed=Positive, Blue=Negative', 
                fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'copula_structure.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved:  {output_dir}/copula_structure.png")
    plt.close()
    
    # Save correlation matrix
    corr_matrix.to_csv(Path(output_dir) / 'copula_correlation_matrix.csv')
    logger.info(f"Saved: {output_dir}/copula_correlation_matrix.csv")
    
    logger.info("✅ Copula visualization complete!")


if __name__ == '__main__': 
    visualize_copula_structure()