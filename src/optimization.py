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
        predictions:  DataFrame com colunas [lower, point, upper] e produtos como index
        n_scenarios: Número de cenários a gerar
        seed: Seed para reprodutibilidade
        
    Returns:  
        Array (n_scenarios, n_products)
    """
    logger.info(f"Gerando {n_scenarios} cenários de demanda")
    
    np.random.seed(seed)
    n_products = len(predictions)
    scenarios = np.zeros((n_scenarios, n_products))
    
    for i, (product, row) in enumerate(predictions.iterrows()):
        lower = row['lower']
        upper = row['upper']
        point = row['point']
        
        # Estimar parâmetros da distribuição
        # Assumindo que point é a média e o intervalo cobre ~90%
        mean = point
        std = (upper - lower) / 3.29  # Para 90% de coverage (~1.645 * 2 * std)
        std = max(std, 0.1)  # Evitar std muito pequeno
        
        # Truncated normal para garantir valores entre [lower, upper]
        if std > 0:
            a = (lower - mean) / std
            b = (upper - mean) / std
            scenarios[: , i] = truncnorm.rvs(a, b, loc=mean, scale=std, size=n_scenarios)
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
            alpha: Percentual dos piores cenários (ex: 0.05 = 5% piores)
            solver: Solver CVXPY a usar
            verbose: Mostrar detalhes da otimização
            
        Returns: 
            Dicionário com alocação ótima e métricas
        """
        logger.info(f"Resolvendo otimização CVaR com alpha={alpha}")
        
        n_scenarios, n_products = scenarios.shape
        
        # Variáveis de decisão
        q = cp.Variable(n_products, nonneg=True)  # Quantidade alocada
        eta = cp.Variable()  # VaR (valor no risco)
        z = cp.Variable(n_scenarios, nonneg=True)  # Excesso sobre VaR
        
        # Calcular custos para cada cenário
        costs = []
        for s in range(n_scenarios):
            demand = scenarios[s, :]
            
            # Custo = custo_falta + custo_excesso
            underage = cp.pos(demand - q)  # max(0, demand - q)
            overage = cp.pos(q - demand)   # max(0, q - demand)
            
            cost_s = self.c_underage * cp.sum(underage) + self.c_overage * cp.sum(overage)
            costs.append(cost_s)
        
        # CVaR = eta + (1/alpha) * E[max(custo - eta, 0)]
        cvar = eta + (1 / (alpha * n_scenarios)) * cp.sum(z)
        
        # Restrições
        constraints = []
        for s in range(n_scenarios):
            constraints.append(z[s] >= costs[s] - eta)
        
        # Problema de otimização
        problem = cp.Problem(cp.Minimize(cvar), constraints)
        
        # Resolver
        try:
            problem.solve(solver=solver, verbose=verbose)
            
            if problem.status not in ['optimal', 'optimal_inaccurate']:
                logger.error(f"Otimização falhou com status: {problem.status}")
                raise ValueError(f"Solver status: {problem.status}")
            
            # Extrair resultados
            allocation = q.value
            cvar_value = cvar.value
            
            # Calcular custo esperado
            expected_cost = np.mean([
                self._calculate_cost(allocation, scenarios[s, :]) 
                for s in range(n_scenarios)
            ])
            
            # Calcular pior caso
            worst_cost = np.max([
                self._calculate_cost(allocation, scenarios[s, :]) 
                for s in range(n_scenarios)
            ])
            
            logger.info("Otimização bem-sucedida")
            logger.info(f"  CVaR: {cvar_value:.2f}")
            logger.info(f"  Custo esperado: {expected_cost:.2f}")
            logger.info(f"  Pior caso: {worst_cost:.2f}")
            
            return {
                'allocation': allocation,
                'cvar':  cvar_value,
                'expected_cost': expected_cost,
                'worst_cost': worst_cost,
                'status':  problem.status
            }
            
        except Exception as e:
            logger.error(f"Erro na otimização: {str(e)}")
            raise
    
    def _calculate_cost(self, allocation:  np.ndarray, demand: np.ndarray) -> float:
        """Calcula custo para uma alocação e demanda específicas."""
        underage = np.maximum(0, demand - allocation)
        overage = np.maximum(0, allocation - demand)
        return self.c_underage * np.sum(underage) + self.c_overage * np.sum(overage)


def evaluate_decisions(
    real_demand: Dict[str, float],
    allocation_cvar: Dict[str, float],
    point_forecasts: Dict[str, float],
    scenarios: np.ndarray,
    c_underage: float = 10.0,
    c_overage: float = 3.0,
    alpha_cvar: float = 0.05
) -> Dict[str, float]:
    """
    Avalia qualidade das decisões de alocação.
    
    Args:
        real_demand: Demanda real por produto
        allocation_cvar: Alocação CVaR por produto
        point_forecasts:  Previsões pontuais por produto
        scenarios:  Cenários de demanda usados
        c_underage:  Custo de falta
        c_overage: Custo de excesso
        alpha_cvar: Alpha usado no CVaR
        
    Returns:
        Dicionário com métricas de decisão
    """
    logger.info("Avaliando decisoes de alocacao")
    
    products = list(real_demand.keys())
    
    # Converter para arrays
    real_demand_array = np.array([real_demand[p] for p in products])
    allocation_cvar_array = np.array([allocation_cvar[p] for p in products])
    point_forecasts_array = np.array([point_forecasts[p] for p in products])
    
    # Função auxiliar de custo
    def calc_cost(allocation, demand):
        underage = np.maximum(0, demand - allocation)
        overage = np.maximum(0, allocation - demand)
        return c_underage * np.sum(underage) + c_overage * np.sum(overage)
    
    # Custo CVaR
    cvar_cost = calc_cost(allocation_cvar_array, real_demand_array)
    
    # Custo baseline (usar previsão pontual)
    mean_cost = calc_cost(point_forecasts_array, real_demand_array)
    
    # Custo Newsvendor (critical fractile)
    critical_fractile = c_underage / (c_underage + c_overage)
    newsvendor_allocation = np.quantile(scenarios, critical_fractile, axis=0)
    newsvendor_cost = calc_cost(newsvendor_allocation, real_demand_array)
    
    # Custo usando quantil alto (conservador)
    quantile_allocation = np.quantile(scenarios, 0.9, axis=0)
    quantile_cost = calc_cost(quantile_allocation, real_demand_array)
    
    # Service level (% da demanda atendida)
    cvar_service_level = 100 * np.minimum(1, allocation_cvar_array / (real_demand_array + 1e-6)).mean()
    
    # Savings
    savings_newsvendor = 100 * (newsvendor_cost - cvar_cost) / (newsvendor_cost + 1e-6)
    savings_mean = 100 * (mean_cost - cvar_cost) / (mean_cost + 1e-6)
    
    logger.info("Metricas de decisao (periodo total):")
    logger.info(f"  Demanda real total: {real_demand_array.sum():.1f}")
    logger.info(f"  Alocacao CVaR total: {allocation_cvar_array.sum():.1f}")
    logger.info(f"\nComparacao por produto:")
    
    for i, product in enumerate(products):
        logger.info(f"  {product}:")
        logger.info(f"    Real: {real_demand[product]:.1f} | CVaR: {allocation_cvar[product]:.1f} | Mean: {point_forecasts[product]:.1f}")
    
    logger.info(f"\nResumo de custos:")
    logger.info(f"  CVaR Cost: {cvar_cost:.2f}")
    logger.info(f"  Mean Baseline Cost: {mean_cost:.2f}")
    logger.info(f"  Newsvendor Cost: {newsvendor_cost:.2f}")
    logger.info(f"  Quantile Cost: {quantile_cost:.2f}")
    logger.info(f"  CVaR Service Level (%): {cvar_service_level:.2f}")
    logger.info(f"  Savings vs Mean (%): {savings_mean:.2f}")
    logger.info(f"  Savings vs Newsvendor (%): {savings_newsvendor:.2f}")
    
    return {
        'cvar_cost':  cvar_cost,
        'mean_cost': mean_cost,
        'newsvendor_cost': newsvendor_cost,
        'quantile_cost': quantile_cost,
        'cvar_service_level': cvar_service_level,
        'savings_mean': savings_mean,
        'savings_newsvendor': savings_newsvendor
    }