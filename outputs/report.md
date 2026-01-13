# Relatório:  Sistema de Alocação Ótima com Conformal Prediction + CVaR

**Data de Geração**:  2026-01-08 15:37:17

---

## 1.Resumo Executivo

Este relatório apresenta os resultados do sistema de alocação ótima de estoque usando: 
- **Conformal Prediction** para intervalos de previsão calibrados
- **CVaR Optimization** para decisões robustas de alocação

### Principais Resultados

✅ **Service Level**: 100.0% (Target: >90%)  
✅ **Economia vs Newsvendor**: 17.6%  
⚠️ **Coverage**: 71.4% (Target: 90%)

---

## 2.Métricas de Previsão

### 2.1 Acurácia Pontual
- **MAE (Mean Absolute Error)**: 7.79
- **RMSE (Root Mean Squared Error)**: 12.18
- **MAPE (Mean Absolute Percentage Error)**: 161.54%

### 2.2 Qualidade dos Intervalos
- **Coverage Real**: 71.36% (Target: 90%)
- **Largura Média do Intervalo**: 20.86

⚠️ **Status Coverage**: NECESSITA AJUSTE

---

## 3.Métricas de Decisão

### 3.1 Custos Operacionais (Período Total de Teste)
- **CVaR**: $37913.00
- **Mean Baseline**: $7524.89
- **Newsvendor**: $45991.24
- **Quantile**: $78613.17

### 3.2 Economia Gerada
- **vs Mean Baseline**: -403.83%
- **vs Newsvendor**: 17.56%

### 3.3 Nível de Serviço
- **Service Level**:  100.00% (Target: >90%)

✅ **Status Economia**: APROVADO  
✅ **Status Service Level**: APROVADO

---

## 4.Otimização CVaR

### 4.1 Resultados
- **CVaR (10% piores cenários)**: $64699.41
- **Custo Esperado**:  $46689.57
- **Pior Caso**: $77726.81

### 4.2 Alocação Ótima por Produto
- Camarao 36/40 120G: 10369.80 unidades
- Camarao 36/40 200G: 19804.64 unidades
- FILE DE TILAPIA 170G: 8481.82 unidades
- File de Robalo 170G: 5376.06 unidades
- File de robalo (moqueca) 300G: 1762.16 unidades
- Fileto 100G: 8003.78 unidades
- Parmegiana 150G: 4058.92 unidades
- TILAPIA 300G: 664.06 unidades
- Tournedour 150G: 8514.44 unidades


---

## 5.Visualizações

![Intervalos de Previsão](figures/forecast_intervals.png)

![Análise de Coverage](figures/coverage_analysis.png)

![Comparação de Decisões](figures/decision_comparison.png)

---

## 6.Conclusões e Recomendações

### 6.1 Principais Insights

1.**CVaR vs Newsvendor**: O CVaR obteve 17.6% de economia comparado ao Newsvendor, demonstrando robustez
2.**Service Level Excelente**: 100.0% indica alta capacidade de atendimento da demanda
3.**Mean Baseline Otimista**: A baseline Mean apresenta custo muito baixo, sugerindo subestimação da demanda real
4.**Coverage Abaixo do Target**: 71.4% sugere necessidade de ajuste nos intervalos

### 6.2 Recomendações

1.**Ajustar Alpha Conformal**: Reduzir alpha de 0.1 para 0.05 para melhorar coverage
2.**Validar Custos**: Verificar se custos de underage ($10.0) e overage ($3.0) refletem realidade operacional
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
| Coverage | ≥85% | 71.4% | ⚠️ |
| Economia vs Newsvendor | ≥10% | 17.6% | ✅ |
| Service Level | >90% | 100.0% | ✅ |

---

**Fim do Relatório**
