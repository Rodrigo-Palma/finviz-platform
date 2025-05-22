# ==============================================
# NOVO CÓDIGO COMPLETO - Forecasting Corrigido e Aprimorado
# ==============================================

import os
import pandas as pd
import numpy as np
import warnings
import pickle
from tqdm import tqdm
from loguru import logger
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from itertools import product

warnings.filterwarnings("ignore")
np.random.seed(42)

# ==========================================
# CONFIGURACAO DO LOGGER E PASTAS
# ==========================================
os.makedirs('logs', exist_ok=True)
os.makedirs('modelos_forecasting', exist_ok=True)
os.makedirs('logs/previsoes', exist_ok=True)
logger.add('logs/forecast_pipeline.log', level='INFO', rotation='10 MB', encoding='utf-8')

# ==========================================
# FUNCOES DE FORECAST
# ==========================================

def treinar_arima(series):
    """Treina um modelo ARIMA(5,1,0) na série fornecida."""
    modelo = ARIMA(series, order=(5, 1, 0))
    modelo_fit = modelo.fit()
    return modelo_fit

def treinar_prophet(df):
    """Treina um modelo Prophet na série fornecida."""
    modelo = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    modelo.fit(df)
    return modelo

def preparar_dados_forecast(caminho_arquivo):
    """Lê e prepara o DataFrame para forecasting."""
    df = pd.read_csv(caminho_arquivo)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['Adj Close'])
    return df

def normalizar_serie(serie):
    """Normaliza a série para o intervalo [0, 1]."""
    min_val = serie.min()
    max_val = serie.max()
    norm = (serie - min_val) / (max_val - min_val)
    return norm, min_val, max_val

def desnormalizar_serie(norm, min_val, max_val):
    """Desfaz a normalização da série."""
    return norm * (max_val - min_val) + min_val

def ajustar_arima(series):
    """Busca automática de ordem ARIMA (p,d,q) com grid simples."""
    best_aic = np.inf
    best_order = (5, 1, 0)
    best_model = None
    # Pequeno grid para tuning
    for p, d, q in product([1,2,3,4,5], [0,1], [0,1,2]):
        try:
            model = ARIMA(series, order=(p, d, q)).fit()
            if model.aic < best_aic:
                best_aic = model.aic
                best_order = (p, d, q)
                best_model = model
        except Exception:
            continue
    return best_model, best_order

def ajustar_prophet(df):
    """Tuning simples de sazonalidade anual/semanal para Prophet."""
    best_mae = np.inf
    best_params = None
    best_model = None
    for yearly in [True, False]:
        for weekly in [True, False]:
            try:
                model = Prophet(daily_seasonality=False, weekly_seasonality=weekly, yearly_seasonality=yearly)
                model.fit(df)
                futuro = model.make_future_dataframe(periods=30)
                forecast = model.predict(futuro)
                mae = mean_absolute_error(df['y'].values[-30:], forecast['yhat'][-30:].values)
                if mae < best_mae:
                    best_mae = mae
                    best_params = (yearly, weekly)
                    best_model = model
            except Exception:
                continue
    return best_model, best_params

def plotar_e_salvar(real, previsto, arquivo, ticker, modelo):
    plt.figure(figsize=(10,4))
    plt.plot(real, label='Real', marker='o')
    plt.plot(previsto, label='Previsto', marker='x')
    plt.title(f'{arquivo} | {ticker} | {modelo}')
    plt.legend()
    plt.tight_layout()
    nome = f'logs/previsoes/plot_{arquivo.replace(",","_").replace(".csv","")}_{ticker}_{modelo}.png'
    plt.savefig(nome)
    plt.close()

def salvar_residuos(real, previsto, arquivo, ticker, modelo):
    residuos = np.array(real) - np.array(previsto)
    df_res = pd.DataFrame({'real': real, 'previsto': previsto, 'residuo': residuos})
    nome = f'logs/previsoes/residuos_{arquivo.replace(",","_").replace(".csv","")}_{ticker}_{modelo}.csv'
    df_res.to_csv(nome, index=False)

# ==========================================
# EXECUCAO PRINCIPAL (TREINAMENTO E SALVAMENTO)
# ==========================================

if __name__ == "__main__":
    logger.info("Iniciando pipeline de forecasting para financas...")

    arquivos = [f for f in os.listdir('dados_transformados') if f.endswith('.csv')]
    metricas = []
    previsoes = []

    for arquivo in tqdm(arquivos, desc="Processando arquivos"):
        try:
            caminho = os.path.join('dados_transformados', arquivo)
            df = preparar_dados_forecast(caminho)

            # Checagem de colunas obrigatórias
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

                    # Normalização opcional
                    norm_series, min_valor, max_valor = normalizar_serie(series)

                    # Tuning ARIMA
                    try:
                        modelo_arima, ordem_arima = ajustar_arima(norm_series)
                        forecast_arima = modelo_arima.forecast(steps=30)
                        pred_arima = desnormalizar_serie(forecast_arima.values[:30], min_valor, max_valor)
                    except Exception as e_arima:
                        logger.exception(f"Erro ARIMA {arquivo} | {ticker}: {e_arima}")
                        modelo_arima = None
                        pred_arima = np.full(30, np.nan)

                    # Tuning Prophet
                    try:
                        df_prophet = df_ticker[['Date', 'Adj Close']].rename(columns={'Date': 'ds', 'Adj Close': 'y'})
                        df_prophet['y_norm'] = norm_series.values
                        modelo_prophet, params_prophet = ajustar_prophet(df_prophet[['ds', 'y_norm']].rename(columns={'y_norm': 'y'}))
                        futuro = modelo_prophet.make_future_dataframe(periods=30)
                        forecast_prophet = modelo_prophet.predict(futuro)
                        pred_prophet = desnormalizar_serie(forecast_prophet['yhat'][-30:].values, min_valor, max_valor)
                    except Exception as e_prophet:
                        logger.exception(f"Erro Prophet {arquivo} | {ticker}: {e_prophet}")
                        modelo_prophet = None
                        pred_prophet = np.full(30, np.nan)

                    real = series[-30:].values
                    if len(real) == 30:
                        if modelo_arima is not None:
                            mse_arima = mean_squared_error(real, pred_arima)
                            mae_arima = mean_absolute_error(real, pred_arima)
                            metricas.append({'arquivo': arquivo, 'ticker': ticker, 'modelo': f'ARIMA{ordem_arima}', 'mse': mse_arima, 'mae': mae_arima})
                            logger.info(f"{arquivo} | {ticker} | ARIMA{ordem_arima} -> MSE: {mse_arima:.6f}, MAE: {mae_arima:.6f}")
                            previsoes.append({'arquivo': arquivo, 'ticker': ticker, 'modelo': f'ARIMA{ordem_arima}', 'real': real.tolist(), 'previsto': pred_arima.tolist()})
                            plotar_e_salvar(real, pred_arima, arquivo, ticker, f'ARIMA{ordem_arima}')
                            salvar_residuos(real, pred_arima, arquivo, ticker, f'ARIMA{ordem_arima}')
                        if modelo_prophet is not None:
                            mse_prophet = mean_squared_error(real, pred_prophet)
                            mae_prophet = mean_absolute_error(real, pred_prophet)
                            metricas.append({'arquivo': arquivo, 'ticker': ticker, 'modelo': f'Prophet{params_prophet}', 'mse': mse_prophet, 'mae': mae_prophet})
                            logger.info(f"{arquivo} | {ticker} | Prophet{params_prophet} -> MSE: {mse_prophet:.6f}, MAE: {mae_prophet:.6f}")
                            previsoes.append({'arquivo': arquivo, 'ticker': ticker, 'modelo': f'Prophet{params_prophet}', 'real': real.tolist(), 'previsto': pred_prophet.tolist()})
                            plotar_e_salvar(real, pred_prophet, arquivo, ticker, f'Prophet{params_prophet}')
                            salvar_residuos(real, pred_prophet, arquivo, ticker, f'Prophet{params_prophet}')

                    # Salvar modelos + min/max
                    if modelo_arima is not None:
                        pickle.dump({'modelo': modelo_arima, 'min': min_valor, 'max': max_valor}, open(f'modelos_forecasting/{ticker}_arima.pkl', 'wb'))
                    if modelo_prophet is not None:
                        pickle.dump({'modelo': modelo_prophet, 'min': min_valor, 'max': max_valor}, open(f'modelos_forecasting/{ticker}_prophet.pkl', 'wb'))

                except Exception as e_ticker:
                    logger.exception(f"Erro ao processar ticker {ticker} em {arquivo}: {e_ticker}")

        except Exception as e:
            logger.exception(f"Erro ao processar {arquivo}: {e}")

    # Salvar métricas e previsões
    pd.DataFrame(metricas).to_csv('logs/previsoes/metricas_forecasting.csv', index=False)
    pd.DataFrame(previsoes).to_csv('logs/previsoes/previsoes_forecasting.csv', index=False)
    logger.info("Pipeline de forecasting finalizado!")