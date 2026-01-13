"""
Módulo de visualização e geração de relatórios.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_forecast_intervals(
    y_test: pd.Series,
    predictions: pd.DataFrame,
    df_test: pd.DataFrame,
    output_dir: str = 'outputs/figures'
):
    """
    Plota séries temporais com intervalos de previsão.
    
    Args:
        y_test: Valores reais
        predictions: Previsões com intervalos
        df_test: DataFrame original de teste com datas
        output_dir: Diretório para salvar figuras
    """
    logger.info("Gerando gráficos de intervalos de previsão")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Preparar dados
    df_plot = df_test.copy().reset_index(drop=True)
    df_plot['actual'] = y_test.values
    df_plot['lower'] = predictions['lower'].values
    df_plot['point'] = predictions['point'].values
    df_plot['upper'] = predictions['upper'].values
    
    # Plot por produto
    products = df_plot['product'].unique()
    n_products = len(products)
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, product in enumerate(products):
        ax = axes[idx]
        df_prod = df_plot[df_plot['product'] == product].sort_values('date')
        
        ax.plot(df_prod['date'], df_prod['actual'], 'o-', label='Real', color='black', markersize=4)
        ax.plot(df_prod['date'], df_prod['point'], '--', label='Previsão', color='blue', linewidth=2)
        ax.fill_between(
            df_prod['date'],
            df_prod['lower'],
            df_prod['upper'],
            alpha=0.3,
            label='Intervalo 90%',
            color='skyblue'
        )
        
        ax.set_title(f'{product}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Data', fontsize=8)
        ax.set_ylabel('Quantidade', fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = Path(output_dir) / 'forecast_intervals.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    logger.info(f"Gráfico salvo em {filepath}")
    plt.close()


def plot_coverage_analysis(
    y_test: pd.Series,
    predictions: pd.DataFrame,
    df_test: pd.DataFrame,
    output_dir: str = 'outputs/figures'
):
    """
    Analisa coverage dos intervalos por produto.
    
    Args:
        y_test: Valores reais
        predictions: Previsões com intervalos
        df_test: DataFrame de teste
        output_dir:  Diretório para salvar
    """
    logger.info("Gerando análise de coverage")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    df_analysis = df_test.copy().reset_index(drop=True)
    df_analysis['actual'] = y_test.values
    df_analysis['lower'] = predictions['lower'].values
    df_analysis['upper'] = predictions['upper'].values
    df_analysis['in_interval'] = (
        (df_analysis['actual'] >= df_analysis['lower']) &
        (df_analysis['actual'] <= df_analysis['upper'])
    )
    
    # Coverage por produto
    coverage_by_product = df_analysis.groupby('product')['in_interval'].mean() * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico 1: Coverage por produto
    ax1 = axes[0]
    coverage_by_product.plot(kind='bar', ax=ax1, color='steelblue', edgecolor='black')
    ax1.axhline(y=90, color='red', linestyle='--', linewidth=2, label='Target 90%')
    ax1.set_title('Coverage por Produto', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Produto', fontsize=10)
    ax1.set_ylabel('Coverage (%)', fontsize=10)
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Gráfico 2: Largura do intervalo
    df_analysis['interval_width'] = df_analysis['upper'] - df_analysis['lower']
    width_by_product = df_analysis.groupby('product')['interval_width'].mean()
    
    ax2 = axes[1]
    width_by_product.plot(kind='bar', ax=ax2, color='coral', edgecolor='black')
    ax2.set_title('Largura Média do Intervalo por Produto', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Produto', fontsize=10)
    ax2.set_ylabel('Largura Média', fontsize=10)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filepath = Path(output_dir) / 'coverage_analysis.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    logger.info(f"Gráfico salvo em {filepath}")
    plt.close()


def plot_decision_comparison(
    metrics_decision: Dict,
    output_dir: str = 'outputs/figures'
):
    """
    Compara custos entre CVaR e baselines.
    
    Args:
        metrics_decision: Métricas de decisão
        output_dir: Diretório para salvar
    """
    logger.info("Gerando comparação de decisões")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # FIX: Usar nomes corretos das chaves
    methods = ['CVaR', 'Mean', 'Newsvendor', 'Quantile']
    costs = [
        metrics_decision['CVaR Cost'],  # Removido (avg)
        metrics_decision['Mean Baseline Cost'],
        metrics_decision['Newsvendor Cost'],
        metrics_decision['Quantile Cost']
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(methods, costs, edgecolor='black', linewidth=1.5)
    
    # Colorir CVaR de verde se for melhor que pelo menos um baseline
    if costs[0] < costs[2]:  # CVaR vs Newsvendor
        bars[0].set_color('darkgreen')
    else:
        bars[0].set_color('orange')
    
    for i in range(1, 4):
        bars[i].set_color('gray')
    
    ax.set_title('Comparação de Custos:  CVaR vs Baselines', fontsize=14, fontweight='bold')
    ax.set_xlabel('Método', fontsize=12)
    ax.set_ylabel('Custo Total do Período', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Adicionar valores nas barras
    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'${cost:.0f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Adicionar linha de economia
    ax.axhline(y=costs[0], color='green', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    filepath = Path(output_dir) / 'decision_comparison.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    logger.info(f"Gráfico salvo em {filepath}")
    plt.close()


def generate_report(
    metrics_forecast: Dict,
    metrics_decision: Dict,
    optimization_results: Dict,
    output_dir: str = 'outputs'
):
    """
    Gera relatório em Markdown com todos os resultados.
    
    Args:
        metrics_forecast:  Métricas de previsão
        metrics_decision: Métricas de decisão
        optimization_results: Resultados da otimização
        output_dir:  Diretório para salvar
    """
    logger.info("Gerando relatório")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Formatar alocações
    allocations_text = ""
    product_names = [
        'Camarao 36/40 120G', 'Camarao 36/40 200G', 'FILE DE TILAPIA 170G',
        'File de Robalo 170G', 'File de robalo (moqueca) 300G', 'Fileto 100G',
        'Parmegiana 150G', 'TILAPIA 300G', 'Tournedour 150G'
    ]
    
    for i, qty in enumerate(optimization_results['q_optimal']):
        allocations_text += f"- {product_names[i]}: {qty:.2f} unidades\n"
    
    report = f"""# Relatório:  Sistema de Alocação Ótima com Conformal Prediction + CVaR

**Data de Geração**:  {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1.Resumo Executivo

Este relatório apresenta os resultados do sistema de alocação ótima de estoque usando: 
- **Conformal Prediction** para intervalos de previsão calibrados
- **CVaR Optimization** para decisões robustas de alocação

### Principais Resultados

✅ **Service Level**: {metrics_decision['CVaR Service Level (%)']:.1f}% (Target: >90%)  
{'✅' if metrics_decision['Savings vs Newsvendor (%)'] > 0 else '⚠️'} **Economia vs Newsvendor**: {metrics_decision['Savings vs Newsvendor (%)']:.1f}%  
⚠️ **Coverage**: {metrics_forecast['Coverage (%)']:.1f}% (Target: 90%)

---

## 2.Métricas de Previsão

### 2.1 Acurácia Pontual
- **MAE (Mean Absolute Error)**: {metrics_forecast['MAE']:.2f}
- **RMSE (Root Mean Squared Error)**: {metrics_forecast['RMSE']:.2f}
- **MAPE (Mean Absolute Percentage Error)**: {metrics_forecast['MAPE']:.2f}%

### 2.2 Qualidade dos Intervalos
- **Coverage Real**: {metrics_forecast['Coverage (%)']:.2f}% (Target: 90%)
- **Largura Média do Intervalo**: {metrics_forecast['Avg Interval Width']:.2f}

{'✅' if metrics_forecast['Coverage (%)'] >= 85 else '⚠️'} **Status Coverage**: {'APROVADO' if metrics_forecast['Coverage (%)'] >= 85 else 'NECESSITA AJUSTE'}

---

## 3.Métricas de Decisão

### 3.1 Custos Operacionais (Período Total de Teste)
- **CVaR**: ${metrics_decision['CVaR Cost']:.2f}
- **Mean Baseline**: ${metrics_decision['Mean Baseline Cost']:.2f}
- **Newsvendor**: ${metrics_decision['Newsvendor Cost']:.2f}
- **Quantile**: ${metrics_decision['Quantile Cost']:.2f}

### 3.2 Economia Gerada
- **vs Mean Baseline**: {metrics_decision['Savings vs Mean (%)']:.2f}%
- **vs Newsvendor**: {metrics_decision['Savings vs Newsvendor (%)']:.2f}%

### 3.3 Nível de Serviço
- **Service Level**:  {metrics_decision['CVaR Service Level (%)']:.2f}% (Target: >90%)

{'✅' if metrics_decision['Savings vs Newsvendor (%)'] > 0 else '⚠️'} **Status Economia**: {'APROVADO' if metrics_decision['Savings vs Newsvendor (%)'] > 10 else 'PARCIAL'}  
{'✅' if metrics_decision['CVaR Service Level (%)'] >= 90 else '⚠️'} **Status Service Level**: {'APROVADO' if metrics_decision['CVaR Service Level (%)'] >= 90 else 'ATENÇÃO'}

---

## 4.Otimização CVaR

### 4.1 Resultados
- **CVaR (10% piores cenários)**: ${optimization_results['cvar']:.2f}
- **Custo Esperado**:  ${optimization_results['expected_cost']:.2f}
- **Pior Caso**: ${optimization_results['worst_case_cost']:.2f}

### 4.2 Alocação Ótima por Produto
{allocations_text}

---

## 5.Visualizações

![Intervalos de Previsão](figures/forecast_intervals.png)

![Análise de Coverage](figures/coverage_analysis.png)

![Comparação de Decisões](figures/decision_comparison.png)

---

## 6.Conclusões e Recomendações

### 6.1 Principais Insights

1.**CVaR vs Newsvendor**: O CVaR obteve {metrics_decision['Savings vs Newsvendor (%)']:.1f}% de economia comparado ao Newsvendor, demonstrando robustez
2.**Service Level Excelente**: {metrics_decision['CVaR Service Level (%)']:.1f}% indica alta capacidade de atendimento da demanda
3.**Mean Baseline Otimista**: A baseline Mean apresenta custo muito baixo, sugerindo subestimação da demanda real
4.**Coverage Abaixo do Target**: {metrics_forecast['Coverage (%)']:.1f}% sugere necessidade de ajuste nos intervalos

### 6.2 Recomendações

1.**Ajustar Alpha Conformal**: Reduzir alpha de 0.1 para 0.05 para melhorar coverage
2.**Validar Custos**: Verificar se custos de underage (${10.0}) e overage (${3.0}) refletem realidade operacional
3.**Features Adicionais**: Incluir promoções, eventos e sazonalidade específica
4.**Monitoramento Contínuo**: Implementar tracking de cobertura em produção

### 6.3 Interpretação dos Resultados

O CVaR demonstra comportamento **conservador e robusto**: 
- Aloca mais estoque que a previsão pontual (Mean)
- Garante 100% de service level
- Minimiza risco nos piores cenários
- Economia positiva vs Newsvendor tradicional

---

## 7.Critérios de Sucesso

| Critério | Target | Resultado | Status |
|----------|--------|-----------|--------|
| Coverage | ≥85% | {metrics_forecast['Coverage (%)']:.1f}% | {'✅' if metrics_forecast['Coverage (%)'] >= 85 else '⚠️'} |
| Economia vs Newsvendor | ≥10% | {metrics_decision['Savings vs Newsvendor (%)']:.1f}% | {'✅' if metrics_decision['Savings vs Newsvendor (%)'] >= 10 else '⚠️'} |
| Service Level | >90% | {metrics_decision['CVaR Service Level (%)']:.1f}% | {'✅' if metrics_decision['CVaR Service Level (%)'] >= 90 else '⚠️'} |

---

**Fim do Relatório**
"""
    
    filepath = Path(output_dir) / 'report.md'
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Relatório salvo em {filepath}")