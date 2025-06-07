import os
import pandas as pd
import numpy as np
import warnings
import pickle
from tqdm import tqdm
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

warnings.filterwarnings("ignore")
np.random.seed(42)

# ========================
# PADRÃO DE SALVAMENTO CSV
# ========================
DEFAULT_TO_CSV_KWARGS = dict(
    sep=';',
    decimal=',',
    index=False
)

# =====================
# CONFIGURACAO DE DIRETORIOS
# =====================
RESULTS_DIR = os.path.join('resultados', 'forecasting_deep')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
RESIDUOS_DIR = os.path.join(RESULTS_DIR, 'residuos')
PREVISOES_DIR = os.path.join(RESULTS_DIR, 'previsoes')
METRICAS_DIR = os.path.join(RESULTS_DIR, 'metricas')
MODELOS_DIR = os.path.join('modelos_forecasting')
LOGS_DIR = 'logs'

for d in [RESULTS_DIR, PLOTS_DIR, RESIDUOS_DIR, PREVISOES_DIR, METRICAS_DIR, MODELOS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

logger.add(os.path.join(LOGS_DIR, 'forecast_pipeline_deep.log'), level='INFO', rotation='10 MB', encoding='utf-8')

# =====================
# FUNCOES DE FORECAST
# =====================

def preparar_dados_forecast(caminho_arquivo):
    df = pd.read_csv(caminho_arquivo, sep=';', decimal=',')
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['Adj Close'])
    return df

def normalizar_serie(serie):
    scaler = MinMaxScaler()
    norm = scaler.fit_transform(serie.values.reshape(-1, 1)).flatten()
    return norm, scaler

def desnormalizar_serie(norm, scaler):
    return scaler.inverse_transform(norm.reshape(-1, 1)).flatten()

# Random Forest

def treinar_rf(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# XGBoost

def treinar_xgb(X_train, y_train):
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# LSTM

def criar_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def preparar_lstm_dados(series, n_steps=10):
    X, y = [], []
    for i in range(len(series) - n_steps):
        X.append(series[i:i+n_steps])
        y.append(series[i+n_steps])
    return np.array(X), np.array(y)

def plotar_e_salvar(real, previsto, arquivo, ticker, modelo):
    plt.figure(figsize=(10,4))
    plt.plot(real, label='Real', marker='o')
    plt.plot(previsto, label='Previsto', marker='x')
    plt.title(f'{arquivo} | {ticker} | {modelo}')
    plt.legend()
    plt.tight_layout()
    nome = os.path.join(PLOTS_DIR, f'plot_{arquivo.replace(",","_").replace(".csv","")}_{ticker}_{modelo}.png')
    plt.savefig(nome)
    plt.close()

def salvar_residuos(real, previsto, arquivo, ticker, modelo):
    residuos = np.array(real) - np.array(previsto)
    df_res = pd.DataFrame({'real': real, 'previsto': previsto, 'residuo': residuos})
    nome = os.path.join(RESIDUOS_DIR, f'residuos_{arquivo.replace(",","_").replace(".csv","")}_{ticker}_{modelo}.csv')
    df_res.to_csv(nome, **DEFAULT_TO_CSV_KWARGS)

# =====================
# EXECUCAO PRINCIPAL
# =====================

if __name__ == "__main__":
    logger.info("Iniciando pipeline de forecasting DEEP (ML models)...")

    arquivos = [f for f in os.listdir('dados_transformados') if f.endswith('.csv')]
    metricas = []
    previsoes = []

    for arquivo in tqdm(arquivos, desc="Processando arquivos"):
        try:
            caminho = os.path.join('dados_transformados', arquivo)
            df = preparar_dados_forecast(caminho)

            if not all(col in df.columns for col in ['Date', 'Ticker', 'Adj Close']):
                logger.warning(f"{arquivo} não possui colunas obrigatórias. Pulando...")
                continue

            for ticker in tqdm(df['Ticker'].unique(), desc=f"{arquivo} - Tickers", leave=False):
                try:
                    logger.info(f"Iniciando {arquivo} | {ticker}")
                    df_ticker = df[df['Ticker'] == ticker].copy()
                    df_ticker.sort_values('Date', inplace=True)
                    series = df_ticker.set_index('Date')['Adj Close']

                    if len(series) < 100:
                        logger.warning(f"Ticker {ticker} com poucos dados. Ignorando...")
                        continue

                    norm_series, scaler = normalizar_serie(series)

                    # Random Forest
                    try:
                        n_lags = 10
                        X_rf = np.array([norm_series[i:i+n_lags] for i in range(len(norm_series)-n_lags-30)])
                        y_rf = np.array([norm_series[i+n_lags] for i in range(len(norm_series)-n_lags-30)])
                        X_rf_train, y_rf_train = X_rf[:-30], y_rf[:-30]
                        X_rf_test = np.array([norm_series[-(30+n_lags)+i:-(30)+i] for i in range(30)])
                        model_rf = treinar_rf(X_rf_train, y_rf_train)
                        pred_rf = model_rf.predict(X_rf_test)
                        pred_rf = desnormalizar_serie(pred_rf, scaler)
                    except Exception as e_rf:
                        logger.exception(f"Erro RF {arquivo} | {ticker}: {e_rf}")
                        model_rf = None
                        pred_rf = np.full(30, np.nan)

                    # XGBoost
                    try:
                        model_xgb = treinar_xgb(X_rf_train, y_rf_train)
                        pred_xgb = model_xgb.predict(X_rf_test)
                        pred_xgb = desnormalizar_serie(pred_xgb, scaler)
                    except Exception as e_xgb:
                        logger.exception(f"Erro XGB {arquivo} | {ticker}: {e_xgb}")
                        model_xgb = None
                        pred_xgb = np.full(30, np.nan)

                    # LSTM
                    try:
                        n_steps = 10
                        X_lstm, y_lstm = preparar_lstm_dados(norm_series, n_steps)
                        X_lstm_train, y_lstm_train = X_lstm[:-30], y_lstm[:-30]
                        X_lstm_test = X_lstm[-30:]
                        model_lstm = criar_lstm((n_steps, 1))
                        es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
                        model_lstm.fit(X_lstm_train.reshape(-1, n_steps, 1), y_lstm_train, epochs=50, batch_size=16, verbose=0, callbacks=[es])
                        pred_lstm = model_lstm.predict(X_lstm_test.reshape(-1, n_steps, 1)).flatten()
                        pred_lstm = desnormalizar_serie(pred_lstm, scaler)
                    except Exception as e_lstm:
                        logger.exception(f"Erro LSTM {arquivo} | {ticker}: {e_lstm}")
                        model_lstm = None
                        pred_lstm = np.full(30, np.nan)

                    real = series[-30:].values
                    if len(real) == 30:
                        # Random Forest
                        if model_rf is not None:
                            mse_rf = mean_squared_error(real, pred_rf)
                            mae_rf = mean_absolute_error(real, pred_rf)
                            metricas.append({'arquivo': arquivo, 'ticker': ticker, 'modelo': f'RandomForest', 'mse': mse_rf, 'mae': mae_rf})
                            logger.info(f"{arquivo} | {ticker} | RandomForest -> MSE: {mse_rf:.6f}, MAE: {mae_rf:.6f}")
                            previsoes.append({'arquivo': arquivo, 'ticker': ticker, 'modelo': f'RandomForest', 'real': real.tolist(), 'previsto': pred_rf.tolist()})
                            plotar_e_salvar(real, pred_rf, arquivo, ticker, f'RandomForest')
                            salvar_residuos(real, pred_rf, arquivo, ticker, f'RandomForest')
                        # XGBoost
                        if model_xgb is not None:
                            mse_xgb = mean_squared_error(real, pred_xgb)
                            mae_xgb = mean_absolute_error(real, pred_xgb)
                            metricas.append({'arquivo': arquivo, 'ticker': ticker, 'modelo': f'XGBoost', 'mse': mse_xgb, 'mae': mae_xgb})
                            logger.info(f"{arquivo} | {ticker} | XGBoost -> MSE: {mse_xgb:.6f}, MAE: {mae_xgb:.6f}")
                            previsoes.append({'arquivo': arquivo, 'ticker': ticker, 'modelo': f'XGBoost', 'real': real.tolist(), 'previsto': pred_xgb.tolist()})
                            plotar_e_salvar(real, pred_xgb, arquivo, ticker, f'XGBoost')
                            salvar_residuos(real, pred_xgb, arquivo, ticker, f'XGBoost')
                        # LSTM
                        if model_lstm is not None:
                            mse_lstm = mean_squared_error(real, pred_lstm)
                            mae_lstm = mean_absolute_error(real, pred_lstm)
                            metricas.append({'arquivo': arquivo, 'ticker': ticker, 'modelo': f'LSTM', 'mse': mse_lstm, 'mae': mae_lstm})
                            logger.info(f"{arquivo} | {ticker} | LSTM -> MSE: {mse_lstm:.6f}, MAE: {mae_lstm:.6f}")
                            previsoes.append({'arquivo': arquivo, 'ticker': ticker, 'modelo': f'LSTM', 'real': real.tolist(), 'previsto': pred_lstm.tolist()})
                            plotar_e_salvar(real, pred_lstm, arquivo, ticker, f'LSTM')
                            salvar_residuos(real, pred_lstm, arquivo, ticker, f'LSTM')

                    # Salvar modelos
                    if model_rf is not None:
                        pickle.dump({'modelo': model_rf, 'scaler': scaler}, open(os.path.join(MODELOS_DIR, f'{ticker}_rf.pkl'), 'wb'))
                    if model_xgb is not None:
                        pickle.dump({'modelo': model_xgb, 'scaler': scaler}, open(os.path.join(MODELOS_DIR, f'{ticker}_xgb.pkl'), 'wb'))
                    if model_lstm is not None:
                        model_lstm.save(os.path.join(MODELOS_DIR, f'{ticker}_lstm.h5'))
                        pickle.dump({'scaler': scaler}, open(os.path.join(MODELOS_DIR, f'{ticker}_lstm_scaler.pkl'), 'wb'))

                except Exception as e_ticker:
                    logger.exception(f"Erro ao processar ticker {ticker} em {arquivo}: {e_ticker}")

        except Exception as e:
            logger.exception(f"Erro ao processar {arquivo}: {e}")

    # Salvar métricas e previsões
    pd.DataFrame(metricas).to_csv(os.path.join(METRICAS_DIR, 'metricas_forecasting_deep.csv'), **DEFAULT_TO_CSV_KWARGS)
    pd.DataFrame(previsoes).to_csv(os.path.join(PREVISOES_DIR, 'previsoes_forecasting_deep.csv'), **DEFAULT_TO_CSV_KWARGS)
    logger.info("Pipeline de forecasting DEEP (ML models) finalizado!")
