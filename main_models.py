"""
Script for comparing multiple forecasting models WITH CVaR optimization.
"""

import logging
import sys
import numpy as np
import pandas as pd
import config

from src.data_preparation import load_data, create_features, split_data
from src.model_comparison import ModelComparison
from src.optimization import generate_scenarios, CVaROptimizer, evaluate_decisions
from src.visualization import plot_model_comparison

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_comparison.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Run multi-model comparison WITH CVaR optimization."""
    
    logger.info("="*80)
    logger.info("MULTI-MODEL COMPARISON PIPELINE (WITH CVAR OPTIMIZATION)")
    logger.info("="*80)
    
    # Set seed
    np.random.seed(config.SEED)
    
    # ========================================================================
    # 1.LOAD DATA
    # ========================================================================
    logger.info("\n📊 Loading and preparing data...")
    
    df = load_data(config.DATA_PATH)
    df_features = create_features(df)
    X_train, y_train, X_cal, y_cal, X_test, y_test = split_data(
        df_features,
        train_end_date=config.TRAIN_END_DATE,
        calibration_end_date=config.CALIBRATION_END_DATE,
        test_start_date=config.TEST_START_DATE,
        calibration_months=config.CALIBRATION_MONTHS
    )
    
    # ✅ FIX: Create df_test with SAME filtering as split_data
    df_test = df_features[df_features['date'] >= pd.to_datetime(config.TEST_START_DATE)].copy()
    df_test = df_test.dropna()  # Same as split_data! 
    df_test = df_test.reset_index(drop=True)  # CRITICAL: Reset index
    
    # ✅ VALIDATE:  Check lengths match
    if len(df_test) != len(X_test):
        logger.error(f"❌ Length mismatch: df_test={len(df_test)}, X_test={len(X_test)}")
        raise ValueError("df_test and X_test must have same length!")
    
    logger.info(f"✅ Data loaded: {len(X_train)} train, {len(X_cal)} cal, {len(X_test)} test")
    
    # ========================================================================
    # 2.INITIALIZE MODEL COMPARISON
    # ========================================================================
    logger.info(f"\n🤖 Initializing {len(config.ENABLED_MODELS)} models...")
    
    comparator = ModelComparison(
        models_config=config.ENABLED_MODELS,
        quantiles=config.QUANTILES,
        seed=config.SEED
    )
    
    # ========================================================================
    # 3.TRAIN ALL MODELS
    # ========================================================================
    comparator.train_all_models(X_train, y_train)
    
    # ========================================================================
    # 4.CALIBRATE CONFORMAL PREDICTORS
    # ========================================================================
    comparator.calibrate_all_models(X_cal, y_cal, alpha=config.ALPHA_CONFORMAL)
    
    # ========================================================================
    # 5.EVALUATE FORECASTING PERFORMANCE
    # ========================================================================
    comparison_df = comparator.evaluate_all_models(X_test, y_test, df_test)
    
        # ========================================================================
    # 6. CVAR OPTIMIZATION FOR EACH MODEL
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("CVAR OPTIMIZATION COMPARISON")
    logger.info("="*80)
    
    cvar_results = {}
    
    # ✅ Get conformal_predictors from comparator
    conformal_predictors = comparator.conformal_predictors
    
    for model_name, cp in conformal_predictors.items():
        try:
            logger.info(f"\n[{model_name}] Running CVaR optimization...")
            
            # Generate prediction intervals
            pred_intervals = cp.predict(X_test)
            
            # ✅ VERIFY: Lengths match
            if len(df_test) != len(pred_intervals):
                logger.error(f"Length mismatch:  df_test={len(df_test)}, predictions={len(pred_intervals)}")
                raise ValueError("Data length mismatch!")
            
            # Get products (sorted for consistency)
            products = sorted(df_test['product'].unique())
            
            # ✅ FIX:  AGGREGATE predictions by product BEFORE generating scenarios
            # This is the KEY fix - we need product-level forecasts, not time-series
            product_forecasts_dict = {}
            for product in products:
                mask = df_test['product'] == product
                product_forecasts_dict[product] = {
                    'mean': pred_intervals.loc[mask, 'point'].sum(),
                    'lower': pred_intervals.loc[mask, 'lower'].sum(),
                    'upper': pred_intervals.loc[mask, 'upper'].sum(),
                    'std': pred_intervals.loc[mask, 'point'].std()  # For scenario generation
                }
            
            # Create aggregated predictions DataFrame for scenario generation
            # Shape: (n_products, 4) instead of (n_timesteps, 3)
            predictions_agg = pd.DataFrame([
                {
                    'product': product,
                    'mean': vals['mean'],
                    'lower': vals['lower'],
                    'upper': vals['upper'],
                    'std': vals['std']
                }
                for product, vals in product_forecasts_dict.items()
            ])
            
            # Generate scenarios based on AGGREGATED product forecasts
            logger.info(f"  Generating {config.N_SCENARIOS} demand scenarios...")
            scenarios = generate_scenarios(
                predictions=predictions_agg,
                n_scenarios=config.N_SCENARIOS,
                seed=config.SEED
            )
            
            # ✅ VALIDATE: Scenarios should be (n_scenarios, n_products)
            if scenarios.shape != (config.N_SCENARIOS, len(products)):
                logger.error(f"Scenario shape mismatch: got {scenarios.shape}, expected ({config.N_SCENARIOS}, {len(products)})")
                raise ValueError(f"Scenarios have wrong shape!")
            
            logger.info(f"  ✅ Scenarios validated: {scenarios.shape}")
            
            # Solve CVaR optimization
            logger.info(f"  Solving CVaR optimization (alpha={config.ALPHA_CVAR})...")
            
            optimizer = CVaROptimizer(
                c_underage=config.C_UNDERAGE,
                c_overage=config.C_OVERAGE,
                seed=config.SEED
            )
            
            result = optimizer.solve(
                scenarios=scenarios,
                alpha=config.ALPHA_CVAR,
                solver=config.CVAR_SOLVER,
                verbose=config.CVAR_VERBOSE
            )
            
            allocation = result['allocation']
            cvar_value = result['cvar']
            expected_cost = result['expected_cost']
            
            # Convert allocation array to dict
            allocation_dict = dict(zip(products, allocation))
            
            # Evaluate allocation decisions
            real_demand = df_test.groupby('product')['quantity'].sum().to_dict()
            point_forecasts = {
                product: product_forecasts_dict[product]['mean']
                for product in products
            }
            
            metrics = evaluate_decisions(
                real_demand=real_demand,
                allocation_cvar=allocation_dict,
                point_forecasts=point_forecasts,
                scenarios=scenarios,
                c_underage=config.C_UNDERAGE,
                c_overage=config.C_OVERAGE,
                alpha_cvar=config.ALPHA_CVAR
            )
            
            # Store results
            cvar_results[model_name] = {
                'cvar_value': cvar_value,
                'expected_cost': expected_cost,
                'actual_cost': metrics['cvar_cost'],
                'service_level': metrics['cvar_service_level'],
                'savings_vs_newsvendor': metrics['savings_newsvendor'],
                'total_allocation': sum(allocation_dict.values()),
                'total_real_demand': sum(real_demand.values())
            }
            
            logger.info(f"  ✅ CVaR:  {cvar_value:.2f}")
            logger.info(f"  ✅ Actual Cost: {metrics['cvar_cost']:.2f}")
            logger.info(f"  ✅ Service Level: {metrics['cvar_service_level']:.2f}%")
            logger.info(f"  ✅ Savings vs Newsvendor: {metrics['savings_newsvendor']:.2f}%")
            
        except Exception as e:
            logger.error(f"  ❌ CVaR optimization failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            cvar_results[model_name] = None
    
    # ========================================================================
    # 7.COMBINE RESULTS
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("COMPREHENSIVE RESULTS (FORECASTING + OPTIMIZATION)")
    logger.info("="*80)
    
    # Add CVaR metrics to comparison dataframe
    if cvar_results:
        for model_name, metrics in cvar_results.items():
            if metrics is not None:
                idx = comparison_df[comparison_df['model'] == model_name].index
                if len(idx) > 0:
                    comparison_df.loc[idx, 'cvar_cost'] = metrics['actual_cost']
                    comparison_df.loc[idx, 'service_level'] = metrics['service_level']
                    comparison_df.loc[idx, 'savings_vs_newsvendor'] = metrics['savings_vs_newsvendor']
        
        # Calculate overall rank including CVaR performance
        if 'cvar_cost' in comparison_df.columns:
            comparison_df['rank_cvar_cost'] = comparison_df['cvar_cost'].rank()
            comparison_df['rank_service_level'] = (100 - abs(comparison_df['service_level'] - 100)).rank(ascending=False)
            
            # New overall rank including CVaR
            comparison_df['overall_rank_with_cvar'] = (
                comparison_df['rank_mae'] + 
                comparison_df['rank_rmse'] + 
                comparison_df['rank_coverage'] +
                comparison_df['rank_width'] +
                comparison_df['rank_cvar_cost'] +
                comparison_df['rank_service_level']
            ) / 6
            
            comparison_df = comparison_df.sort_values('overall_rank_with_cvar')
    
    # ========================================================================
    # 8.DISPLAY COMPREHENSIVE RESULTS
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("FINAL RANKING (FORECASTING + CVAR OPTIMIZATION)")
    logger.info("="*80)
    
    if 'overall_rank_with_cvar' in comparison_df.columns:
        display_cols = ['model', 'mae', 'coverage', 'cvar_cost', 'service_level', 
                       'savings_vs_newsvendor', 'overall_rank_with_cvar']
        display_df = comparison_df[display_cols].copy()
        
        # Format
        display_df['mae'] = display_df['mae'].apply(lambda x: f"{x:.2f}")
        display_df['coverage'] = display_df['coverage'].apply(lambda x: f"{x:.2f}%")
        display_df['cvar_cost'] = display_df['cvar_cost'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        display_df['service_level'] = display_df['service_level'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
        display_df['savings_vs_newsvendor'] = display_df['savings_vs_newsvendor'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
        display_df['overall_rank_with_cvar'] = display_df['overall_rank_with_cvar'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        
        logger.info(f"\n{display_df.to_string(index=False)}")
        
        best_model = comparison_df.iloc[0]['model']
        logger.info(f"\n🏆 BEST MODEL (Overall): {best_model}")
    else:
        comparator.print_comparison_table()
    
    # ========================================================================
    # 9.SAVE RESULTS
    # ========================================================================
    comparator.save_results(config.RESULTS_DIR)
    comparison_df.to_csv(config.RESULTS_DIR / 'model_comparison_with_cvar.csv', index=False)
    logger.info(f"CVaR results saved to {config.RESULTS_DIR / 'model_comparison_with_cvar.csv'}")
    
    # ========================================================================
    # 10.VISUALIZE COMPARISON
    # ========================================================================
    logger.info("\n📈 Generating comparison visualizations...")
    
    try:
        plot_model_comparison(comparison_df, config.FIGURES_DIR)
    except Exception as e:
        logger.warning(f"Visualization failed: {e}")
    
    logger.info("\n" + "="*80)
    logger.info("✅ COMPREHENSIVE MODEL COMPARISON COMPLETED!")
    logger.info("="*80)
    logger.info(f"\nResults saved in:")
    logger.info(f"  - {config.RESULTS_DIR / 'model_comparison_with_cvar.csv'}")
    logger.info(f"  - {config.FIGURES_DIR}")
    

if __name__ == '__main__':
    main()