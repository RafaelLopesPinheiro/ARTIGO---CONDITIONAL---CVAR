"""
Módulo de preparação de dados para o sistema de alocação ótima.
Responsável por carregar, transformar e preparar dados para modelagem.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """
    Carrega dados de vendas do arquivo CSV.
    
    Args:
        filepath: Caminho para o arquivo CSV
        
    Returns:
        DataFrame no formato long (date, product, quantity)
    """
    logger.info(f"Carregando dados de {filepath}")
    
    # Carregar CSV com formato brasileiro de datas
    df_wide = pd.read_csv(filepath)
    
    # Parsear datas no formato DD/MM/YYYY
    df_wide['date'] = pd.to_datetime(df_wide['date'], format='%d/%m/%Y')
    
    # Transformar de wide para long
    product_cols = [col for col in df_wide.columns if col != 'date']
    df_long = df_wide.melt(
        id_vars=['date'],
        value_vars=product_cols,
        var_name='product',
        value_name='quantity'
    )
    
    # Ordenar por produto e data
    df_long = df_long.sort_values(['product', 'date']).reset_index(drop=True)
    
    logger.info(f"Dados carregados:  {len(df_long)} observações, {df_long['product'].nunique()} produtos")
    logger.info(f"Período: {df_long['date'].min()} a {df_long['date'].max()}")
    
    return df_long


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features temporais e de produto para modelagem.
    
    Args:
        df: DataFrame com colunas [date, product, quantity]
        
    Returns:
        DataFrame com features adicionadas
    """
    logger.info("Criando features")
    
    df = df.copy()
    
    # Features temporais
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    
    # Encoding de produto
    df['product_id'] = pd.Categorical(df['product']).codes
    
    # Features de lags e rolling por produto
    features_list = []
    
    for product in df['product'].unique():
        df_prod = df[df['product'] == product].copy().sort_values('date')
        
        # Lags
        df_prod['lag_1'] = df_prod['quantity'].shift(1)
        df_prod['lag_7'] = df_prod['quantity'].shift(7)
        df_prod['lag_30'] = df_prod['quantity'].shift(30)
        
        # Rolling statistics
        df_prod['mean_7'] = df_prod['quantity'].shift(1).rolling(window=7, min_periods=1).mean()
        df_prod['std_7'] = df_prod['quantity'].shift(1).rolling(window=7, min_periods=1).std()
        df_prod['mean_30'] = df_prod['quantity'].shift(1).rolling(window=30, min_periods=1).mean()
        df_prod['std_30'] = df_prod['quantity'].shift(1).rolling(window=30, min_periods=1).std()
        
        # Min/Max recentes
        df_prod['min_7'] = df_prod['quantity'].shift(1).rolling(window=7, min_periods=1).min()
        df_prod['max_7'] = df_prod['quantity'].shift(1).rolling(window=7, min_periods=1).max()
        
        # Zero count (para demanda intermitente)
        df_prod['zeros_7'] = (df_prod['quantity'].shift(1).rolling(window=7, min_periods=1).apply(
            lambda x: (x == 0).sum(), raw=True))
        
        features_list.append(df_prod)
    
    df_features = pd.concat(features_list, ignore_index=True)
    
    # Preencher NaN com zeros para início da série
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns
    df_features[numeric_cols] = df_features[numeric_cols].fillna(0)
    
    logger.info(f"Features criadas: {len(df_features.columns)} colunas")
    
    return df_features


def split_data(
    df: pd.DataFrame,
    train_end_date: str = '2024-12-31',
    test_start_date: str = '2025-01-01',
    calibration_months: int = 2
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Divide dados em treino, calibração e teste.
    
    Args:
        df: DataFrame com features
        train_end_date: Data final do treino (formato YYYY-MM-DD)
        test_start_date: Data inicial do teste
        calibration_months: Meses para calibração (últimos do treino)
        
    Returns: 
        X_train, y_train, X_cal, y_cal, X_test, y_test
    """
    logger.info("Dividindo dados em treino, calibração e teste")
    
    # Converter datas
    train_end = pd.to_datetime(train_end_date)
    test_start = pd.to_datetime(test_start_date)
    cal_start = train_end - pd.DateOffset(months=calibration_months)
    
    # Features para modelagem
    feature_cols = [
        'day_of_week', 'is_weekend', 'day_of_month', 'month',
        'product_id', 'lag_1', 'lag_7', 'lag_30',
        'mean_7', 'std_7', 'mean_30', 'std_30',
        'min_7', 'max_7', 'zeros_7'
    ]
    
    # Filtrar dados válidos (após período de warmup)
    df_valid = df[df['lag_30'].notna()].copy()
    
    # Divisão treino/calibração/teste
    train_mask = df_valid['date'] <= train_end
    cal_mask = (df_valid['date'] > cal_start) & (df_valid['date'] <= train_end)
    test_mask = df_valid['date'] >= test_start
    
    X_train = df_valid.loc[train_mask, feature_cols]
    y_train = df_valid.loc[train_mask, 'quantity']
    
    X_cal = df_valid.loc[cal_mask, feature_cols]
    y_cal = df_valid.loc[cal_mask, 'quantity']
    
    X_test = df_valid.loc[test_mask, feature_cols]
    y_test = df_valid.loc[test_mask, 'quantity']
    
    logger.info(f"Treino: {len(X_train)} obs({df_valid.loc[train_mask, 'date'].min()} a {df_valid.loc[train_mask, 'date'].max()})")
    logger.info(f"Calibração: {len(X_cal)} obs({df_valid.loc[cal_mask, 'date'].min()} a {df_valid.loc[cal_mask, 'date'].max()})")
    logger.info(f"Teste: {len(X_test)} obs({df_valid.loc[test_mask, 'date'].min()} a {df_valid.loc[test_mask, 'date'].max()})")
    
    return X_train, y_train, X_cal, y_cal, X_test, y_test