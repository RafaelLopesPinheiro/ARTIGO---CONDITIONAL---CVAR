"""
Módulo de otimização de alocação usando CVaR (Conditional Value at Risk).
Minimiza custos no pior cenário para decisões robustas.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import cvxpy as cp
from scipy.stats import truncnorm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_scenarios(
    predictions: pd.DataFrame,
    n_scenarios: int = 1000,
    seed: int = 42
) -> np.ndarray:
    """
    Gera cenários de demanda a partir de intervalos de previsão.
    
    Args:
        predictions: DataFrame com [lower, point, upper]
        n_scenarios: Número de cenários a gerar
        seed: Seed para reprodutibilidade
        
    Returns: 
        Array (n_scenarios, n_products)
    """
    logger.info(f"Gerando {n_scenarios} cenários de demanda")
    
    np.random.seed(seed)
    n_products = len(predictions)
    scenarios = np.zeros((n_scenarios, n_products))
    
    for i in range(n_products):
        lower = predictions.iloc[i]['lower']
        upper = predictions.iloc[i]['upper']
        point = predictions.iloc[i]['point']
        
        # Estimar parâmetros da distribuição
        # Assumindo que point é a média e o intervalo cobre ~90%
        mean = point
        std = (upper - lower) / 3.29  # Para 90% de coverage (~1.645 * 2 * std)
        std = max(std, 0.1)  # Evitar std muito pequeno
        
        # Truncated normal para garantir valores entre [lower, upper]
        if std > 0:
            a = (lower - mean) / std
            b = (upper - mean) / std
            scenarios[:, i] = truncnorm.rvs(a, b, loc=mean, scale=std, size=n_scenarios)
        else:
            scenarios[:, i] = np.full(n_scenarios, mean)
        
        # Garantir não-negatividade
        scenarios[:, i] = np.maximum(0, scenarios[:, i])
    
    logger.info(f"Cenários gerados: shape {scenarios.shape}")
    
    return scenarios


class CVaROptimizer:
    """
    Otimizador baseado em CVaR para alocação robusta de estoque.
    """
    
    def __init__(
        self,
        c_underage: float = 10.0,
        c_overage: float = 3.0,
        seed: int = 42
    ):
        """
        Args:
            c_underage:  Custo de falta de estoque (por unidade)
            c_overage: Custo de excesso de estoque (por unidade)
            seed: Seed para reprodutibilidade
        """
        self.c_underage = c_underage
        self.c_overage = c_overage
        self.seed = seed
        self.q_optimal = None
        self.optimal_cost = None
        
    def solve(
        self,
        scenarios: np.ndarray,
        alpha: float = 0.1,
        solver: str = 'ECOS',
        verbose: bool = False
    ) -> Dict:
        """
        Resolve problema de otimização CVaR.
        
        Args:
            scenarios: Array (n_scenarios, n_products) de demanda
            alpha: Nível de CVaR (0.1 = 10% piores cenários)
            solver: Solver CVXPY ('ECOS', 'SCS', 'CVXOPT')
            verbose: Se True, exibe informações do solver
            
        Returns: 
            Dicionário com resultados da otimização
        """
        logger.info(f"Resolvendo otimização CVaR com alpha={alpha}")
        
        n_scenarios, n_products = scenarios.shape
        
        # Variáveis de decisão
        q = cp.Variable(n_products, nonneg=True)  # Quantidade a alocar
        eta = cp.Variable()  # VaR
        zeta = cp.Variable(n_scenarios, nonneg=True)  # Desvios acima do VaR
        
        # Custos por cenário
        costs = []
        for s in range(n_scenarios):
            underage = cp.maximum(scenarios[s, :] - q, 0)
            overage = cp.maximum(q - scenarios[s, :], 0)
            cost_s = self.c_underage * cp.sum(underage) + self.c_overage * cp.sum(overage)
            costs.append(cost_s)
        
        # Restrições de CVaR
        constraints = []
        for s in range(n_scenarios):
            constraints.append(zeta[s] >= costs[s] - eta)
        
        # Objetivo: minimizar CVaR
        objective = cp.Minimize(
            eta + (1 / (alpha * n_scenarios)) * cp.sum(zeta)
        )
        
        # Resolver
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=solver, verbose=verbose)
            
            if problem.status == cp.OPTIMAL: 
                self.q_optimal = q.value
                self.optimal_cost = problem.value
                
                # Calcular métricas adicionais
                expected_cost = np.mean([
                    self._calculate_cost(self.q_optimal, scenarios[s, :])
                    for s in range(n_scenarios)
                ])
                
                worst_case_cost = np.max([
                    self._calculate_cost(self.q_optimal, scenarios[s, :])
                    for s in range(n_scenarios)
                ])
                
                results = {
                    'q_optimal': self.q_optimal,
                    'cvar': self.optimal_cost,
                    'expected_cost': expected_cost,
                    'worst_case_cost': worst_case_cost,
                    'status': problem.status
                }
                
                logger.info(f"Otimização bem-sucedida")
                logger.info(f"  CVaR: {self.optimal_cost:.2f}")
                logger.info(f"  Custo esperado: {expected_cost:.2f}")
                logger.info(f"  Pior caso: {worst_case_cost:.2f}")
                
                return results
            else:
                logger.error(f"Otimização falhou: {problem.status}")
                raise ValueError(f"Optimization failed with status: {problem.status}")
                
        except Exception as e:
            logger.error(f"Erro na otimização: {str(e)}")
            raise
    
    def _calculate_cost(self, q: np.ndarray, demand: np.ndarray) -> float:
        """Calcula custo para uma alocação e demanda específica."""
        underage = np.maximum(demand - q, 0)
        overage = np.maximum(q - demand, 0)
        return self.c_underage * np.sum(underage) + self.c_overage * np.sum(overage)


def evaluate_decisions(
    optimization_results: Dict,
    y_test: pd.Series,
    predictions: pd.DataFrame,
    df_test: pd.DataFrame,
    c_underage:  float = 10.0,
    c_overage: float = 3.0
) -> Dict:
    """
    Avalia decisões de alocação comparando com baselines.
    
    Args:
        optimization_results: Resultados da otimização CVaR
        y_test: Demanda real de teste
        predictions: Previsões com intervalos
        df_test: DataFrame de teste com informações de produto
        c_underage: Custo de falta
        c_overage: Custo de excesso
        
    Returns:
        Dicionário com métricas comparativas
    """
    logger.info("Avaliando decisoes de alocacao")
    
    q_cvar = optimization_results['q_optimal']
    
    # FIX: Reset de índices para garantir alinhamento
    df_test_reset = df_test.reset_index(drop=True)
    y_test_reset = y_test.reset_index(drop=True)
    predictions_reset = predictions.reset_index(drop=True)
    
    # Preparar dados
    products = df_test_reset['product'].unique()
    n_products = len(products)
    
    # Agregar demanda REAL por produto (soma do período)
    y_actual_by_product = []
    predictions_by_product_list = []
    
    for product in products:
        mask = df_test_reset['product'] == product
        y_actual_by_product.append(y_test_reset[mask].sum())
        predictions_by_product_list.append({
            'lower': predictions_reset[mask]['lower'].sum(),
            'point':  predictions_reset[mask]['point'].sum(),
            'upper': predictions_reset[mask]['upper'].sum()
        })
    
    y_actual = np.array(y_actual_by_product)
    predictions_agg = pd.DataFrame(predictions_by_product_list)
    
    # Calcular custos para cada método
    
    # 1.CVaR
    underage_cvar = np.maximum(y_actual - q_cvar, 0)
    overage_cvar = np.maximum(q_cvar - y_actual, 0)
    cost_cvar = c_underage * np.sum(underage_cvar) + c_overage * np.sum(overage_cvar)
    service_level_cvar = np.mean(y_actual <= q_cvar)
    
    # 2.Mean Baseline (point prediction)
    q_mean = predictions_agg['point'].values
    underage_mean = np.maximum(y_actual - q_mean, 0)
    overage_mean = np.maximum(q_mean - y_actual, 0)
    cost_mean = c_underage * np.sum(underage_mean) + c_overage * np.sum(overage_mean)
    
    # 3.Newsvendor (critical fractile)
    critical_fractile = c_underage / (c_underage + c_overage)
    q_news = predictions_agg['lower'].values + \
             critical_fractile * (predictions_agg['upper'].values - predictions_agg['lower'].values)
    underage_news = np.maximum(y_actual - q_news, 0)
    overage_news = np.maximum(q_news - y_actual, 0)
    cost_news = c_underage * np.sum(underage_news) + c_overage * np.sum(overage_news)
    
    # 4.Quantile direto (upper)
    q_quantile = predictions_agg['upper'].values
    underage_q = np.maximum(y_actual - q_quantile, 0)
    overage_q = np.maximum(q_quantile - y_actual, 0)
    cost_q = c_underage * np.sum(underage_q) + c_overage * np.sum(overage_q)
    
    metrics = {
        'CVaR Cost':  cost_cvar,
        'Mean Baseline Cost': cost_mean,
        'Newsvendor Cost': cost_news,
        'Quantile Cost': cost_q,
        'CVaR Service Level (%)': service_level_cvar * 100,
        'Savings vs Mean (%)': (1 - cost_cvar / cost_mean) * 100 if cost_mean > 0 else 0,
        'Savings vs Newsvendor (%)': (1 - cost_cvar / cost_news) * 100 if cost_news > 0 else 0,
    }
    
    logger.info("Metricas de decisao (periodo total):")
    logger.info(f"  Demanda real total: {y_actual.sum():.1f}")
    logger.info(f"  Alocacao CVaR total: {q_cvar.sum():.1f}")
    
    # Log detalhado por produto
    logger.info("\nComparacao por produto:")
    for i, product in enumerate(products):
        logger.info(f"  {product}:")
        logger.info(f"    Real: {y_actual[i]:.1f} | CVaR: {q_cvar[i]:.1f} | Mean: {q_mean[i]:.1f}")
    
    logger.info("\nResumo de custos:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.2f}")
    
    return metrics