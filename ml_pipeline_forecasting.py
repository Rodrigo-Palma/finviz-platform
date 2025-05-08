# ==============================================
# NOVO CÓDIGO COMPLETO - Forecasting Corrigido
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

warnings.filterwarnings("ignore")

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
    modelo = ARIMA(series, order=(5, 1, 0))
    modelo_fit = modelo.fit()
    return modelo_fit

def treinar_prophet(df):
    modelo = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    modelo.fit(df)
    return modelo

def preparar_dados_forecast(caminho_arquivo):
    df = pd.read_csv(caminho_arquivo)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna(subset=['Adj Close'])
    return df

# ==========================================
# EXECUCAO PRINCIPAL (TREINAMENTO E SALVAMENTO)
# ==========================================

if __name__ == "__main__":
    logger.info("Iniciando pipeline de forecasting para financas...")

    arquivos = [f for f in os.listdir('dados_transformados') if f.endswith('.csv')]

    for arquivo in tqdm(arquivos, desc="Processando arquivos"):
        try:
            caminho = os.path.join('dados_transformados', arquivo)
            df = preparar_dados_forecast(caminho)

            for ticker in df['Ticker'].unique():
                df_ticker = df[df['Ticker'] == ticker].copy()
                df_ticker.sort_values('Date', inplace=True)

                series = df_ticker.set_index('Date')['Adj Close']

                if len(series) < 100:
                    logger.warning(f"Ticker {ticker} com poucos dados. Ignorando...")
                    continue

                min_valor = series.min()
                max_valor = series.max()

                # Treinamento ARIMA
                modelo_arima = treinar_arima(series)

                # Treinamento Prophet
                df_prophet = df_ticker[['Date', 'Adj Close']].rename(columns={'Date': 'ds', 'Adj Close': 'y'})
                modelo_prophet = treinar_prophet(df_prophet)

                # Avaliação de performance
                forecast_arima = modelo_arima.forecast(steps=30)
                futuro = modelo_prophet.make_future_dataframe(periods=30)
                forecast_prophet = modelo_prophet.predict(futuro)

                real = series[-30:].values
                pred_arima = forecast_arima.values[:30] if len(forecast_arima) >= 30 else forecast_arima.values
                pred_prophet = forecast_prophet['yhat'][-30:].values

                if len(real) == len(pred_arima) == len(pred_prophet):
                    mse_arima = mean_squared_error(real, pred_arima)
                    mae_arima = mean_absolute_error(real, pred_arima)

                    mse_prophet = mean_squared_error(real, pred_prophet)
                    mae_prophet = mean_absolute_error(real, pred_prophet)

                    logger.info(f"{arquivo} | {ticker} | ARIMA -> MSE: {mse_arima:.6f}, MAE: {mae_arima:.6f}")
                    logger.info(f"{arquivo} | {ticker} | Prophet -> MSE: {mse_prophet:.6f}, MAE: {mae_prophet:.6f}")

                # Salvar modelos + min/max
                pickle.dump({'modelo': modelo_arima, 'min': min_valor, 'max': max_valor}, open(f'modelos_forecasting/{ticker}_arima.pkl', 'wb'))
                pickle.dump({'modelo': modelo_prophet, 'min': min_valor, 'max': max_valor}, open(f'modelos_forecasting/{ticker}_prophet.pkl', 'wb'))

        except Exception as e:
            logger.exception(f"Erro ao processar {arquivo}: {e}")

    logger.info("Pipeline de forecasting finalizado!")