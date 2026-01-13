"""
Configurações centralizadas do Sistema de Alocação Ótima.
Permite fácil ajuste de parâmetros e experimentação.
"""

from pathlib import Path

# ============================================================================
# CONFIGURAÇÕES GERAIS
# ============================================================================

# Seed para reprodutibilidade
SEED = 42

# Diretórios
DATA_PATH = 'data/vendas.csv'
OUTPUT_DIR = 'outputs'
FIGURES_DIR = Path(OUTPUT_DIR) / 'figures'
RESULTS_DIR = Path(OUTPUT_DIR) / 'results'
LOG_FILE = 'pipeline.log'

# ============================================================================
# CONFIGURAÇÕES DE DADOS
# ============================================================================

# Divisão temporal
TRAIN_END_DATE = '2024-12-31'
TEST_START_DATE = '2025-01-01'
CALIBRATION_MONTHS = 2  # Últimos N meses do treino para calibração

# ============================================================================
# CONFORMAL PREDICTION
# ============================================================================

# Alpha: nível de significância (1-alpha = coverage desejado)
# Alpha = 0.10 → Coverage 90%
# Alpha = 0.05 → Coverage 95%
ALPHA_CONFORMAL = 0.15  # AJUSTADO para melhorar coverage

# Quantis para previsão
QUANTILES = [0.1, 0.5, 0.9]

# ============================================================================
# LIGHTGBM - QUANTILE FORECASTING
# ============================================================================

LGBM_PARAMS = {
    'objective': 'quantile',  # Será sobrescrito para cada quantil
    'n_estimators': 200,
    'max_depth': 5,
    'learning_rate':  0.05,
    'num_leaves': 31,
    'min_child_samples':  20,
    'subsample': 0.8,
    'colsample_bytree':  0.8,
    'random_state': SEED,
    'verbose': -1
}

# ============================================================================
# OTIMIZAÇÃO CVaR
# ============================================================================

# Alpha CVaR: percentual dos piores cenários a considerar
# Alpha = 0.10 → Otimiza para 10% piores cenários
# Alpha = 0.05 → Otimiza para 5% piores cenários (mais conservador)
ALPHA_CVAR = 0.1

# Número de cenários de demanda a gerar
N_SCENARIOS = 1000

# Custos operacionais (em R$)
C_UNDERAGE = 10.0  # Custo de falta de estoque (por unidade)
C_OVERAGE = 3.0    # Custo de excesso de estoque (por unidade)

# Solver CVaR
CVAR_SOLVER = 'ECOS'  # Opções: 'ECOS', 'SCS', 'CVXOPT'
CVAR_VERBOSE = False

# ============================================================================
# VISUALIZAÇÃO
# ============================================================================

# Configurações de gráficos
PLOT_DPI = 300
PLOT_STYLE = 'whitegrid'
FIGURE_SIZE_WIDE = (18, 12)
FIGURE_SIZE_STANDARD = (12, 6)
FIGURE_SIZE_COMPARISON = (14, 5)

# ============================================================================
# FEATURES
# ============================================================================

# Features temporais
TEMPORAL_FEATURES = [
    'day_of_week',
    'is_weekend',
    'day_of_month',
    'month'
]

# Features de produto
PRODUCT_FEATURES = [
    'product_id'
]

# Features de lags
LAG_FEATURES = [1, 7, 30]

# Features de rolling statistics
ROLLING_WINDOWS = {
    'short':  7,   # Janela curta
    'long': 30    # Janela longa
}

# ============================================================================
# NOMES DOS PRODUTOS (para relatórios)
# ============================================================================

PRODUCT_NAMES = [
    'Camarao 36/40 120G',
    'Camarao 36/40 200G',
    'FILE DE TILAPIA 170G',
    'File de Robalo 170G',
    'File de robalo (moqueca) 300G',
    'Fileto 100G',
    'Parmegiana 150G',
    'TILAPIA 300G',
    'Tournedour 150G'
]

# ============================================================================
# CRITÉRIOS DE SUCESSO
# ============================================================================

TARGET_COVERAGE = 85.0      # Coverage mínimo aceitável (%)
TARGET_SERVICE_LEVEL = 90.0  # Service level mínimo (%)
TARGET_SAVINGS = 10.0        # Economia mínima vs baseline (%)

# ============================================================================
# CONFIGURAÇÕES AVANÇADAS
# ============================================================================

# Logging
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Encoding
FILE_ENCODING = 'utf-8'

# Validação
VALIDATE_INPUTS = True
RAISE_ON_LOW_COVERAGE = False  # Se True, levanta erro se coverage < target

# ============================================================================
# CONFIGURAÇÕES EXPERIMENTAIS
# ============================================================================

# Análise de sensibilidade (descomentar para ativar)
SENSITIVITY_ANALYSIS = False

# Diferentes cenários de custo para testar
COST_SCENARIOS = [
    {'name': 'Base', 'c_underage': 10.0, 'c_overage': 3.0},
    {'name': 'High Shortage', 'c_underage':  15.0, 'c_overage': 3.0},
    {'name':  'Balanced', 'c_underage':  7.0, 'c_overage': 5.0},
]

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def get_feature_columns():
    """Retorna lista de todas as features para modelagem."""
    features = TEMPORAL_FEATURES + PRODUCT_FEATURES
    
    # Adicionar lags
    for lag in LAG_FEATURES:
        features.append(f'lag_{lag}')
    
    # Adicionar rolling statistics
    for window_name, window_size in ROLLING_WINDOWS.items():
        features.extend([
            f'mean_{window_size}',
            f'std_{window_size}',
            f'min_{window_size}',
            f'max_{window_size}',
        ])
    
    # Adicionar zero count
    features.append(f'zeros_{ROLLING_WINDOWS["short"]}')
    
    return features


def validate_config():
    """Valida configurações antes da execução."""
    errors = []
    
    # Validar alpha
    if not 0 < ALPHA_CONFORMAL < 1:
        errors.append(f"ALPHA_CONFORMAL deve estar entre 0 e 1, recebido: {ALPHA_CONFORMAL}")
    
    if not 0 < ALPHA_CVAR < 1:
        errors.append(f"ALPHA_CVAR deve estar entre 0 e 1, recebido: {ALPHA_CVAR}")
    
    # Validar custos
    if C_UNDERAGE <= 0:
        errors.append(f"C_UNDERAGE deve ser positivo, recebido: {C_UNDERAGE}")
    
    if C_OVERAGE <= 0:
        errors.append(f"C_OVERAGE deve ser positivo, recebido: {C_OVERAGE}")
    
    # Validar paths
    if not Path(DATA_PATH).exists():
        errors.append(f"Arquivo de dados não encontrado: {DATA_PATH}")
    
    # Validar quantis
    if not all(0 < q < 1 for q in QUANTILES):
        errors.append(f"Todos os quantis devem estar entre 0 e 1: {QUANTILES}")
    
    if errors:
        raise ValueError("Erros de configuração encontrados:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True


def print_config():
    """Imprime configurações principais."""
    print("\n" + "="*80)
    print("CONFIGURAÇÕES DO SISTEMA")
    print("="*80)
    print(f"Seed: {SEED}")
    print(f"Data:  {DATA_PATH}")
    print(f"\nConformal Prediction:")
    print(f"  - Alpha: {ALPHA_CONFORMAL} (Coverage target: {(1-ALPHA_CONFORMAL)*100:.0f}%)")
    print(f"  - Quantis: {QUANTILES}")
    print(f"\nCVaR Optimization:")
    print(f"  - Alpha:  {ALPHA_CVAR} (Piores {ALPHA_CVAR*100:.0f}% cenários)")
    print(f"  - Cenários: {N_SCENARIOS}")
    print(f"  - Custo falta: R$ {C_UNDERAGE:.2f}")
    print(f"  - Custo excesso: R$ {C_OVERAGE:.2f}")
    print(f"\nCritérios de Sucesso:")
    print(f"  - Coverage: ≥{TARGET_COVERAGE:.0f}%")
    print(f"  - Service Level: ≥{TARGET_SERVICE_LEVEL:.0f}%")
    print(f"  - Savings: ≥{TARGET_SAVINGS:.0f}%")
    print("="*80 + "\n")


if __name__ == "__main__": 
    # Validar configurações
    try:
        validate_config()
        print_config()
        print("✅ Configurações validadas com sucesso!")
    except ValueError as e:
        print(f"❌ {e}")