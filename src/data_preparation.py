# src/data_preparation.py

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def load_data(file_path:  str) -> pd.DataFrame:
    """Load sales data from CSV."""
    logger.info(f"Carregando dados de {file_path}")
    
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    
    # Convert to long format
    product_cols = [col for col in df.columns if col != 'date']
    df_long = df.melt(
        id_vars=['date'],
        value_vars=product_cols,
        var_name='product',
        value_name='quantity'
    )
    
    df_long = df_long.sort_values(['product', 'date']).reset_index(drop=True)
    
    logger.info(f"Dados carregados:  {len(df_long)} observações, {df_long['product'].nunique()} produtos")
    logger.info(f"Período:  {df_long['date'].min()} a {df_long['date'].max()}")
    
    return df_long


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time series features for demand forecasting."""
    logger.info("Criando features")
    
    df = df.copy()
    df = df.sort_values(['product', 'date']).reset_index(drop=True)
    
    # Time-based features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['weekofyear'] = df['date'].dt.isocalendar().week
    
    # Cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # Lag features (per product)
    for lag in [7, 14, 30]:
        df[f'lag_{lag}'] = df.groupby('product')['quantity'].shift(lag)
    
    # Rolling statistics (per product)
    for window in [7, 14, 30]:
        df[f'rolling_mean_{window}'] = df.groupby('product')['quantity'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f'rolling_std_{window}'] = df.groupby('product')['quantity'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std()
        )
    
    logger.info(f"Features criadas: {df.shape[1]} colunas")
    
    return df


def split_data(
    df_features: pd.DataFrame,
    train_end_date: str,
    calibration_end_date: str,  # NEW PARAMETER
    test_start_date: str,
    calibration_months: int = 2
):
    """
    Split data into train, calibration, and test sets WITHOUT LEAKAGE.
    
    CRITICAL FIX: Calibration set is NOW SEPARATE from training set.
    
    Args:
        df_features: DataFrame with features
        train_end_date: End date for training (e.g., '2024-08-31')
        calibration_end_date: End date for calibration (e.g., '2024-12-31')
        test_start_date: Start date for test (e.g., '2025-01-01')
        calibration_months: Number of months for calibration
        
    Returns:
        X_train, y_train, X_cal, y_cal, X_test, y_test
    """
    logger.info("Dividindo dados em treino, calibração e teste")
    
    # ========================================================================
    # FIXED SPLIT: NO OVERLAP BETWEEN TRAIN AND CALIBRATION
    # ========================================================================
    
    # Training set: Up to train_end_date (e.g., Jan-Aug 2024)
    df_train = df_features[df_features['date'] <= pd.to_datetime(train_end_date)].copy()
    
    # Calibration set: Between train_end and calibration_end (e.g., Sept-Dec 2024)
    # This is SEPARATE from training! 
    df_cal = df_features[
        (df_features['date'] > pd.to_datetime(train_end_date)) &
        (df_features['date'] <= pd.to_datetime(calibration_end_date))
    ].copy()
    
    # Test set: From test_start onwards (e.g., Jan 2025+)
    df_test = df_features[df_features['date'] >= pd.to_datetime(test_start_date)].copy()
    
    # Remove rows with missing lag features
    df_train = df_train.dropna()
    df_cal = df_cal.dropna()
    df_test = df_test.dropna()
    
    # Log split info
    logger.info(f"Treino: {len(df_train)} obs({df_train['date'].min()} a {df_train['date'].max()})")
    logger.info(f"Calibração:  {len(df_cal)} obs({df_cal['date'].min()} a {df_cal['date'].max()})")
    logger.info(f"Teste: {len(df_test)} obs({df_test['date'].min()} a {df_test['date'].max()})")
    
    # Check for overlap (should be NONE)
    train_dates = set(df_train['date'].unique())
    cal_dates = set(df_cal['date'].unique())
    test_dates = set(df_test['date'].unique())
    
    if train_dates & cal_dates: 
        logger.error(f"❌ DATA LEAKAGE: Training and calibration overlap by {len(train_dates & cal_dates)} dates!")
        raise ValueError("Training and calibration sets must not overlap!")
    
    if train_dates & test_dates: 
        logger.error(f"❌ DATA LEAKAGE: Training and test overlap by {len(train_dates & test_dates)} dates!")
        raise ValueError("Training and test sets must not overlap!")
        
    if cal_dates & test_dates: 
        logger.error(f"❌ DATA LEAKAGE: Calibration and test overlap by {len(cal_dates & test_dates)} dates!")
        raise ValueError("Calibration and test sets must not overlap!")
    
    logger.info("✅ No data leakage detected - splits are independent")
    
    # Prepare features and targets
    feature_cols = [col for col in df_train.columns if col not in ['date', 'product', 'quantity']]
    
    X_train = df_train[feature_cols]
    y_train = df_train['quantity']
    
    X_cal = df_cal[feature_cols]
    y_cal = df_cal['quantity']
    
    X_test = df_test[feature_cols]
    y_test = df_test['quantity']
    
    return X_train, y_train, X_cal, y_cal, X_test, y_test


def validate_splits(X_train, y_train, X_cal, y_cal, X_test, y_test, df_features, config):
    """
    Validate that data splits are consistent and have no index mismatches. 
    """
    logger.info("\n🔍 Validating data splits...")
    
    # Check lengths
    assert len(X_train) == len(y_train), "Train X and y length mismatch!"
    assert len(X_cal) == len(y_cal), "Cal X and y length mismatch!"
    assert len(X_test) == len(y_test), "Test X and y length mismatch!"
    
    # Check df_test recreation matches
    df_test_recreated = df_features[
        df_features['date'] >= pd.to_datetime(config.TEST_START_DATE)
    ].copy().dropna().reset_index(drop=True)
    
    if len(df_test_recreated) != len(X_test):
        logger.error(f"❌ df_test recreation mismatch: {len(df_test_recreated)} vs {len(X_test)}")
        raise ValueError("df_test recreation produces different length!")
    
    logger.info(f"✅ All splits validated successfully")
    logger.info(f"   Train: {len(X_train)}, Cal: {len(X_cal)}, Test: {len(X_test)}")
    
    return True
