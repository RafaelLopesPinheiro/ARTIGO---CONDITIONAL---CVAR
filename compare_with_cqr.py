"""
Compare Split Conformal vs CQR (Conformalized Quantile Regression).
Shows improvement from asymmetric calibration.
"""

import logging
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import config
from src.data_preparation import load_data, create_features, split_data
from src.forecasting import QuantileForecaster, evaluate_forecast
from src.conformal_product_specific import ProductSpecificConformalPredictor
from src.conformal_cqr import ProductSpecificCQR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comparison_cqr.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def calculate_coverage_metrics(
    y_true:pd.Series,
    predictions:pd.DataFrame,
    products:pd.Series
) -> dict:
    """Calculate comprehensive coverage metrics."""
    
    df = pd.DataFrame({
        'y_true':y_true.values,
        'lower': predictions['lower'].values,
        'upper':predictions['upper'].values,
        'product':products.values
    })
    
    # Overall metrics
    df['covered'] = (df['y_true'] >= df['lower']) & (df['y_true'] <= df['upper'])
    df['width'] = df['upper'] - df['lower']
    
    # Per-product
    product_metrics = []
    for product in sorted(df['product'].unique()):
        mask = df['product'] == product
        product_metrics.append({
            'product':product,
            'coverage_%':df[mask]['covered'].mean() * 100,
            'mean_width':df[mask]['width'].mean(),
            'n_samples':mask.sum()
        })
    
    # Joint coverage (all products covered at same time)
    unique_products = df['product'].unique()
    n_products = len(unique_products)
    n_dates = len(df) // n_products
    
    df['date_idx'] = np.repeat(np.arange(n_dates), n_products)[:len(df)]
    joint_coverage = df.groupby('date_idx')['covered'].all().mean() * 100
    
    return {
        'overall_coverage_%':df['covered'].mean() * 100,
        'joint_coverage_%':joint_coverage,
        'mean_width':df['width'].mean(),
        'median_width':df['width'].median(),
        'product_metrics':pd.DataFrame(product_metrics)
    }


def visualize_cqr_comparison(
    results_split:dict,
    results_cqr:dict,
    output_dir:str = 'outputs/cqr_comparison'
):
    """Visualize comparison between Split Conformal and CQR."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    target_coverage = (1 - config.ALPHA_CONFORMAL) * 100
    
    # 1.Coverage Comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Overall coverage
    ax = axes[0]
    methods = ['Split\nConformal', 'CQR']
    coverages = [
        results_split['overall_coverage_%'],
        results_cqr['overall_coverage_%']
    ]
    
    bars = ax.bar(methods, coverages, alpha=0.8, color=['steelblue', 'coral'], edgecolor='black', linewidth=2)
    ax.axhline(target_coverage, color='red', linestyle='--', linewidth=2, label=f'Target ({target_coverage:.0f}%)')
    ax.set_ylabel('Coverage (%)', fontsize=12)
    ax.set_title('Overall Coverage', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])
    
    # Add values on bars
    for bar, val in zip(bars, coverages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Joint coverage
    ax = axes[1]
    joint_coverages = [
        results_split['joint_coverage_%'],
        results_cqr['joint_coverage_%']
    ]
    
    bars = ax.bar(methods, joint_coverages, alpha=0.8, color=['steelblue', 'coral'], edgecolor='black', linewidth=2)
    ax.set_ylabel('Joint Coverage (%)', fontsize=12)
    ax.set_title('Joint Coverage (All SKUs)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])
    
    for bar, val in zip(bars, joint_coverages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Interval width
    ax = axes[2]
    widths = [
        results_split['mean_width'],
        results_cqr['mean_width']
    ]
    
    bars = ax.bar(methods, widths, alpha=0.8, color=['steelblue', 'coral'], edgecolor='black', linewidth=2)
    ax.set_ylabel('Mean Interval Width', fontsize=12)
    ax.set_title('Interval Efficiency', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, widths):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Add improvement annotation
    improvement = results_cqr['overall_coverage_%'] - results_split['overall_coverage_%']
    width_change = ((results_cqr['mean_width'] - results_split['mean_width']) / results_split['mean_width']) * 100
    
    fig.text(0.5, 0.02, 
             f'CQR Improvement:Coverage {improvement:+.1f} pp  |  Width {width_change:+.1f}%',
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgreen' if improvement > 0 else 'lightyellow', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(Path(output_dir) / 'cqr_vs_split_comparison.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_dir}/cqr_vs_split_comparison.png")
    plt.close()
    
    # 2.Per-product comparison
    df_split = results_split['product_metrics']
    df_cqr = results_cqr['product_metrics']
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    products = df_split['product'].values
    x = np.arange(len(products))
    width = 0.35
    
    ax.bar(x - width/2, df_split['coverage_%'], width, label='Split Conformal', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, df_cqr['coverage_%'], width, label='CQR', alpha=0.8, color='coral')
    
    ax.axhline(target_coverage, color='red', linestyle='--', linewidth=2, label=f'Target ({target_coverage:.0f}%)')
    
    ax.set_ylabel('Coverage (%)', fontsize=12)
    ax.set_title('Per-Product Coverage: Split Conformal vs CQR', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([p[:20] + '...' if len(p) > 20 else p for p in products], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'per_product_coverage_comparison.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved:{output_dir}/per_product_coverage_comparison.png")
    plt.close()


def main():
    """Compare Split Conformal vs CQR."""
    
    logger.info("="*80)
    logger.info("SPLIT CONFORMAL vs CQR COMPARISON")
    logger.info("="*80)
    
    np.random.seed(config.SEED)
    
    # Load filtered data
    logger.info("\nLoading filtered data...")
    try:
        df = load_data('data/vendas.csv')
    except FileNotFoundError:
        logger.error("Filtered dataset not found!  Run filter_skus.py first.")
        return
    
    df_features = create_features(df)
    X_train, y_train, X_cal, y_cal, X_test, y_test = split_data(
        df_features,
        train_end_date=config.TRAIN_END_DATE,
        test_start_date=config.TEST_START_DATE,
        calibration_months=config.CALIBRATION_MONTHS
    )
    
    df_test = df_features[df_features['date'] >= pd.to_datetime(config.TEST_START_DATE)].copy()
    df_test = df_test[df_test['lag_30'].notna()].reset_index(drop=True)
    
    products_cal = df_features[df_features.index.isin(X_cal.index)]['product']
    products_test = df_test['product']
    
    logger.info(f"Data splits:")
    logger.info(f"  Train:{len(X_train)}")
    logger.info(f"  Calibration:{len(X_cal)} ({len(products_cal.unique())} products)")
    logger.info(f"  Test:{len(X_test)} ({len(products_test.unique())} products)")
    
    # Train forecaster
    logger.info("\n" + "="*80)
    logger.info("TRAINING QUANTILE FORECASTER")
    logger.info("="*80)
    
    qf = QuantileForecaster(
        quantiles=config.QUANTILES,
        seed=config.SEED,
        params=config.LGBM_PARAMS
    )
    qf.fit(X_train, y_train)
    
    # Method 1:Split Conformal (Product-Specific)
    logger.info("\n" + "="*80)
    logger.info("METHOD 1:SPLIT CONFORMAL (Product-Specific)")
    logger.info("="*80)
    
    cp_split = ProductSpecificConformalPredictor(seed=config.SEED)
    cp_split.calibrate(qf, X_cal, y_cal, products_cal, alpha=config.ALPHA_CONFORMAL)
    
    preds_split = cp_split.predict(X_test, products_test)
    metrics_split = calculate_coverage_metrics(y_test, preds_split, products_test)
    
    logger.info(f"\nSplit Conformal Results:")
    logger.info(f"  Overall Coverage:{metrics_split['overall_coverage_%']:.2f}%")
    logger.info(f"  Joint Coverage: {metrics_split['joint_coverage_%']:.2f}%")
    logger.info(f"  Mean Width:{metrics_split['mean_width']:.3f}")
    
    # Method 2:CQR (Product-Specific)
    logger.info("\n" + "="*80)
    logger.info("METHOD 2:CQR (Product-Specific)")
    logger.info("="*80)
    
    cqr = ProductSpecificCQR(seed=config.SEED)
    cqr.calibrate(qf, X_cal, y_cal, products_cal, alpha=config.ALPHA_CONFORMAL)
    
    preds_cqr = cqr.predict(X_test, products_test)
    metrics_cqr = calculate_coverage_metrics(y_test, preds_cqr, products_test)
    
    logger.info(f"\nCQR Results:")
    logger.info(f"  Overall Coverage:{metrics_cqr['overall_coverage_%']:.2f}%")
    logger.info(f"  Joint Coverage:{metrics_cqr['joint_coverage_%']:.2f}%")
    logger.info(f"  Mean Width:{metrics_cqr['mean_width']:.3f}")
    
    # Show CQR corrections
    logger.info("\nCQR Asymmetric Corrections:")
    corrections_df = cqr.get_corrections_summary()
    logger.info(f"\n{corrections_df.to_string(index=False)}")
    
    # Comparison
    logger.info("\n" + "="*80)
    logger.info("COMPARISON")
    logger.info("="*80)
    
    cov_improvement = metrics_cqr['overall_coverage_%'] - metrics_split['overall_coverage_%']
    width_change = ((metrics_cqr['mean_width'] - metrics_split['mean_width']) / metrics_split['mean_width']) * 100
    
    logger.info(f"\nCQR vs Split Conformal:")
    logger.info(f"  Coverage Improvement:{cov_improvement:+.2f} pp")
    logger.info(f"  Width Change: {width_change:+.2f}%")
    logger.info(f"  Joint Coverage Improvement:{metrics_cqr['joint_coverage_%'] - metrics_split['joint_coverage_%']:+.2f} pp")
    
    # Visualizations
    visualize_cqr_comparison(metrics_split, metrics_cqr, output_dir='outputs/cqr_comparison')
    
    # Save results
    summary = pd.DataFrame({
        'Method':['Split_Conformal', 'CQR'],
        'Overall_Coverage_%':[metrics_split['overall_coverage_%'], metrics_cqr['overall_coverage_%']],
        'Joint_Coverage_%':[metrics_split['joint_coverage_%'], metrics_cqr['joint_coverage_%']],
        'Mean_Width':[metrics_split['mean_width'], metrics_cqr['mean_width']],
        'Median_Width': [metrics_split['median_width'], metrics_cqr['median_width']]
    })
    
    Path('outputs/cqr_comparison').mkdir(parents=True, exist_ok=True)
    summary.to_csv('outputs/cqr_comparison/cqr_comparison_summary.csv', index=False)
    logger.info("\nSaved: outputs/cqr_comparison/cqr_comparison_summary.csv")
    
    # Final verdict
    logger.info("\n" + "="*80)
    logger.info("FINAL VERDICT")
    logger.info("="*80)
    
    target = (1 - config.ALPHA_CONFORMAL) * 100
    
    if abs(metrics_cqr['overall_coverage_%'] - target) < abs(metrics_split['overall_coverage_%'] - target):
        logger.info("\n🏆 CQR WINS: Better coverage calibration!")
    elif cov_improvement > 2:
        logger.info("\n🏆 CQR WINS: Significantly better coverage!")
    elif abs(cov_improvement) < 1 and abs(width_change) < 2:
        logger.info("\n🤝 TIE:Both methods perform similarly")
    else:
        logger.info("\n✅ Split Conformal sufficient")
    
    logger.info("\n📊 Check outputs/cqr_comparison/ for visualizations")
    logger.info("="*80)


if __name__ == '__main__':
    main()