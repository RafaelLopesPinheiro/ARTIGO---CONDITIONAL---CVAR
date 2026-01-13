"""
Script principal do Sistema de Alocação Ótima.
Orquestra todo o pipeline:  preparação, previsão, otimização e visualização.
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Importar configurações
import config

# Configurar logging COM ENCODING UTF-8
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding=config.FILE_ENCODING),
        logging.StreamHandler(sys.stdout)
    ]
)

# FIX: Configurar stdout para UTF-8 no Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logger = logging.getLogger(__name__)

# Importar módulos
from src.data_preparation import load_data, create_features, split_data
from src.forecasting import QuantileForecaster, ConformalPredictor, evaluate_forecast
from src.optimization import generate_scenarios, CVaROptimizer, evaluate_decisions
from src.visualization import (
    plot_forecast_intervals,
    plot_coverage_analysis,
    plot_decision_comparison,
    generate_report
)


def main():
    """Executa pipeline completo."""
    
    logger.info("="*80)
    logger.info("INICIANDO PIPELINE DE ALOCACAO OTIMA")
    logger.info("="*80)
    
    # Validar e exibir configurações
    try:
        config.validate_config()
        config.print_config()
    except ValueError as e:
        logger.error(f"Erro de configuração: {e}")
        return
    
    # Definir seed global
    np.random.seed(config.SEED)
    
    # ========================================================================
    # 1.PREPARAÇÃO DE DADOS
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("ETAPA 1: PREPARACAO DE DADOS")
    logger.info("="*80)
    
    try:
        df = load_data(config.DATA_PATH)
        df_features = create_features(df)
        X_train, y_train, X_cal, y_cal, X_test, y_test = split_data(
            df_features,
            train_end_date=config.TRAIN_END_DATE,
            test_start_date=config.TEST_START_DATE,
            calibration_months=config.CALIBRATION_MONTHS
        )
        
        # Guardar informações de teste para visualização
        df_test = df_features[df_features['date'] >= pd.to_datetime(config.TEST_START_DATE)].copy()
        df_test = df_test[df_test['lag_30'].notna()].reset_index(drop=True)
        
        logger.info("[OK] Preparacao de dados concluida com sucesso")
        
    except Exception as e:
        logger.error(f"[ERRO] Erro na preparacao de dados: {str(e)}")
        raise
    
    # ========================================================================
    # 2.PREVISÃO COM CONFORMAL PREDICTION
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("ETAPA 2: PREVISAO PROBABILISTICA")
    logger.info("="*80)
    
    try:
        # Treinar quantile forecaster
        qf = QuantileForecaster(
            quantiles=config.QUANTILES,
            seed=config.SEED,
            params=config.LGBM_PARAMS
        )
        qf.fit(X_train, y_train)
        
        # Calibrar conformal predictor
        cp = ConformalPredictor(seed=config.SEED)
        cp.calibrate(qf, X_cal, y_cal, alpha=config.ALPHA_CONFORMAL)
        
        # Prever no conjunto de teste
        predictions = cp.predict(X_test)
        
        # Avaliar previsões
        metrics_forecast = evaluate_forecast(y_test, predictions)
        
        # Verificar coverage
        if config.RAISE_ON_LOW_COVERAGE and metrics_forecast['Coverage (%)'] < config.TARGET_COVERAGE:
            logger.warning(
                f"Coverage ({metrics_forecast['Coverage (%)']:.1f}%) abaixo do target "
                f"({config.TARGET_COVERAGE:.1f}%)"
            )
        
        logger.info("[OK] Previsao concluida com sucesso")
        
    except Exception as e: 
        logger.error(f"[ERRO] Erro na previsao: {str(e)}")
        raise
    
    # ========================================================================
    # 3.OTIMIZAÇÃO CVaR
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("ETAPA 3: OTIMIZACAO DE ALOCACAO")
    logger.info("="*80)
    
    try:
        # Agregar previsões por produto (SOMA do período de teste)
        products = df_test['product'].unique()
        n_products = len(products)
        
        predictions_by_product = []
        for product in products: 
            mask = df_test['product'] == product
            pred_prod = {
                'lower': predictions[mask]['lower'].sum(),
                'point': predictions[mask]['point'].sum(),
                'upper': predictions[mask]['upper'].sum()
            }
            predictions_by_product.append(pred_prod)
        
        predictions_agg = pd.DataFrame(predictions_by_product)
        
        logger.info(f"Previsoes agregadas por produto (periodo total):")
        for i, product in enumerate(products):
            logger.info(f"  {product}: point={predictions_agg.iloc[i]['point']:.1f}")
        
        # Gerar cenários
        scenarios = generate_scenarios(
            predictions_agg,
            n_scenarios=config.N_SCENARIOS,
            seed=config.SEED
        )
        
        # Otimizar com CVaR
        optimizer = CVaROptimizer(
            c_underage=config.C_UNDERAGE,
            c_overage=config.C_OVERAGE,
            seed=config.SEED
        )
        
        optimization_results = optimizer.solve(
            scenarios,
            alpha=config.ALPHA_CVAR,
            solver=config.CVAR_SOLVER,
            verbose=config.CVAR_VERBOSE
        )
        
        # Avaliar decisões
        metrics_decision = evaluate_decisions(
            optimization_results,
            y_test,
            predictions,
            df_test,
            c_underage=config.C_UNDERAGE,
            c_overage=config.C_OVERAGE
        )
        
        logger.info("[OK] Otimizacao concluida com sucesso")
        
    except Exception as e:
        logger.error(f"[ERRO] Erro na otimizacao: {str(e)}")
        raise
    
    # ========================================================================
    # 4.VISUALIZAÇÃO E RELATÓRIOS
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("ETAPA 4: GERACAO DE VISUALIZACOES E RELATORIOS")
    logger.info("="*80)
    
    try:
        # Criar diretórios de saída
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Gerar gráficos
        plot_forecast_intervals(y_test, predictions, df_test, output_dir=str(config.FIGURES_DIR))
        plot_coverage_analysis(y_test, predictions, df_test, output_dir=str(config.FIGURES_DIR))
        plot_decision_comparison(metrics_decision, output_dir=str(config.FIGURES_DIR))
        
        # Salvar resultados em CSV
        predictions_df = predictions.copy()
        predictions_df['actual'] = y_test.values
        predictions_df['product'] = df_test['product'].values
        predictions_df['date'] = df_test['date'].values
        predictions_df.to_csv(config.RESULTS_DIR / 'predictions.csv', index=False)
        
        allocations_df = pd.DataFrame({
            'product':  products,
            'optimal_allocation': optimization_results['q_optimal']
        })
        allocations_df.to_csv(config.RESULTS_DIR / 'allocations.csv', index=False)
        
        metrics_df = pd.DataFrame([{**metrics_forecast, **metrics_decision}])
        metrics_df.to_csv(config.RESULTS_DIR / 'forecast_metrics.csv', index=False)
        
        # Gerar relatório
        generate_report(
            metrics_forecast,
            metrics_decision,
            optimization_results,
            output_dir=str(config.OUTPUT_DIR)
        )
        
        logger.info("[OK] Visualizacoes e relatorios gerados com sucesso")
        
    except Exception as e:
        logger.error(f"[ERRO] Erro na visualizacao: {str(e)}")
        raise
    
    # ========================================================================
    # FINALIZAÇÃO E VALIDAÇÃO
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("VALIDACAO DOS RESULTADOS")
    logger.info("="*80)
    
    # Verificar critérios de sucesso
    success_criteria = {
        'Coverage':  (metrics_forecast['Coverage (%)'], config.TARGET_COVERAGE, '>='),
        'Service Level':  (metrics_decision['CVaR Service Level (%)'], config.TARGET_SERVICE_LEVEL, '>='),
        'Savings vs Newsvendor': (metrics_decision['Savings vs Newsvendor (%)'], 0, '>='),
    }
    
    all_passed = True
    for criterion, (value, target, operator) in success_criteria.items():
        if operator == '>=': 
            passed = value >= target
        else: 
            passed = value <= target
        
        status = "[OK]" if passed else "[ATENCAO]"
        logger.info(f"{status} {criterion}:  {value:.2f} (target: {operator}{target:.2f})")
        
        if not passed:
            all_passed = False
    
    # ========================================================================
    # CONCLUSÃO
    # ========================================================================
    logger.info("\n" + "="*80)
    if all_passed:
        logger.info("PIPELINE CONCLUIDO COM SUCESSO!  TODOS OS CRITERIOS ATENDIDOS!")
    else:
        logger.info("PIPELINE CONCLUIDO!  ALGUNS CRITERIOS NECESSITAM ATENCAO.")
    logger.info("="*80)
    logger.info("\nResultados disponiveis em:")
    logger.info(f"  - {config.FIGURES_DIR}/")
    logger.info(f"  - {config.RESULTS_DIR}/")
    logger.info(f"  - {config.OUTPUT_DIR}/report.md")
    logger.info(f"\nLog completo salvo em: {config.LOG_FILE}")
    

if __name__ == "__main__":
    main()