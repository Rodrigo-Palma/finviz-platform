import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from loguru import logger

# =============================
# CONFIGURAÇÃO DO LOGGER
# =============================
os.makedirs('logs', exist_ok=True)
logger.add('logs/engenharia_avancada.log', level='INFO', rotation='10 MB', encoding='utf-8')

# =============================
# GARANTE PASTAS
# =============================
def garantir_pastas():
    os.makedirs('dados_transformados', exist_ok=True)

# =============================
# FUNÇÕES DE INDICADORES
# =============================

def calcular_moving_averages(df, periods=[5, 20, 50, 100, 200]):
    for period in periods:
        df[f'SMA_{period}'] = df.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(window=period).mean())
        df[f'EMA_{period}'] = df.groupby('Ticker')['Adj Close'].transform(lambda x: x.ewm(span=period, adjust=False).mean())
    return df

def calcular_rsi(df, period=14):
    delta = df.groupby('Ticker')['Adj Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.groupby(df['Ticker']).transform(lambda x: x.rolling(window=period).mean())
    avg_loss = loss.groupby(df['Ticker']).transform(lambda x: x.rolling(window=period).mean())
    rs = avg_gain / avg_loss
    df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
    return df

def calcular_macd(df):
    ema_12 = df.groupby('Ticker')['Adj Close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema_26 = df.groupby('Ticker')['Adj Close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df.groupby('Ticker')['MACD'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    return df

def calcular_bollinger_bands(df, period=20):
    sma = df.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(window=period).mean())
    std = df.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(window=period).std())
    df['Bollinger_Upper'] = sma + (std * 2)
    df['Bollinger_Lower'] = sma - (std * 2)
    df['Bollinger_Width'] = (df['Bollinger_Upper'] - df['Bollinger_Lower']) / sma
    return df

def calcular_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df.groupby('Ticker')['Close'].shift())
    low_close = np.abs(df['Low'] - df.groupby('Ticker')['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.groupby(df['Ticker']).transform(lambda x: x.rolling(window=period).mean())
    return df

def calcular_obv(df):
    if 'Volume' not in df.columns:
        logger.warning("Coluna 'Volume' não encontrada. Pulando cálculo de OBV.")
        df['OBV'] = 0
        return df

    obv = []
    for ticker, grupo in df.groupby('Ticker'):
        obv_ticker = [0]
        for i in range(1, len(grupo)):
            if grupo['Adj Close'].iloc[i] > grupo['Adj Close'].iloc[i-1]:
                obv_ticker.append(obv_ticker[-1] + grupo['Volume'].iloc[i])
            elif grupo['Adj Close'].iloc[i] < grupo['Adj Close'].iloc[i-1]:
                obv_ticker.append(obv_ticker[-1] - grupo['Volume'].iloc[i])
            else:
                obv_ticker.append(obv_ticker[-1])
        obv.extend(obv_ticker)
    df['OBV'] = obv
    return df

def calcular_retorno_volatilidade(df):
    df['Return_1d'] = df.groupby('Ticker')['Adj Close'].ffill().pct_change(1)
    df['Return_5d'] = df.groupby('Ticker')['Adj Close'].ffill().pct_change(5)
    df['Return_20d'] = df.groupby('Ticker')['Adj Close'].ffill().pct_change(20)
    df['Volatility_5d'] = df.groupby('Ticker')['Return_1d'].transform(lambda x: x.rolling(window=5).std())
    df['Volatility_20d'] = df.groupby('Ticker')['Return_1d'].transform(lambda x: x.rolling(window=20).std())
    return df

def calcular_sharpe_ratio(df, risk_free_rate=0.02):
    daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1
    df['Excess_Return'] = df['Return_1d'] - daily_risk_free
    df['Sharpe_20d'] = df.groupby('Ticker')['Excess_Return'].transform(lambda x: x.rolling(window=20).mean() / x.rolling(window=20).std()) * np.sqrt(252)
    return df

def criar_lags(df, colunas, lags=[1, 2, 3, 5]):
    for col in colunas:
        if col not in df.columns:
            logger.warning(f"Coluna '{col}' não encontrada para criação de LAGs. Pulando...")
            continue
        for lag in lags:
            df[f'{col}_lag{lag}'] = df.groupby('Ticker')[col].shift(lag)
    return df

# =============================
# PROCESSAMENTO PRINCIPAL
# =============================

def calcular_atributos_completos(df):
    df = calcular_moving_averages(df)
    df = calcular_rsi(df)
    df = calcular_macd(df)
    df = calcular_bollinger_bands(df)
    df = calcular_atr(df)
    df = calcular_obv(df)
    df = calcular_retorno_volatilidade(df)
    df = calcular_sharpe_ratio(df)

    colunas_para_lags = ['Adj Close', 'Volume', 'Return_1d', 'MACD', 'RSI_14']
    df = criar_lags(df, colunas=colunas_para_lags)

    return df

if __name__ == "__main__":
    garantir_pastas()
    logger.info("Iniciando engenharia de atributos avançada incremental com LAGs...")

    arquivos = [f for f in os.listdir('dados') if f.endswith('.csv')]

    for arquivo in tqdm(arquivos, desc="Processando arquivos"):
        try:
            caminho_arquivo = os.path.join('dados', arquivo)
            df_novo = pd.read_csv(caminho_arquivo)
            if 'Date' in df_novo.columns:
                df_novo['Date'] = pd.to_datetime(df_novo['Date'])

            caminho_saida = os.path.join('dados_transformados', arquivo)
            if os.path.exists(caminho_saida):
                df_existente = pd.read_csv(caminho_saida)
                if 'Date' in df_existente.columns:
                    df_existente['Date'] = pd.to_datetime(df_existente['Date'])
                df_merged = pd.concat([df_existente, df_novo]).drop_duplicates(subset=['Date', 'Ticker']).reset_index(drop=True)
            else:
                df_merged = df_novo

            if 'Ticker' not in df_merged.columns or 'Adj Close' not in df_merged.columns:
                logger.warning(f"Arquivo ignorado: {arquivo} (faltam colunas obrigatórias)")
                continue

            df_transformado = calcular_atributos_completos(df_merged)
            df_transformado.to_csv(caminho_saida, index=False)
            logger.info(f"Arquivo incremental salvo: {caminho_saida}")

        except Exception as e:
            logger.exception(f"Erro processando {arquivo}: {e}")

    logger.info("Engenharia de atributos incremental concluída! Arquivos salvos em 'dados_transformados'.")
