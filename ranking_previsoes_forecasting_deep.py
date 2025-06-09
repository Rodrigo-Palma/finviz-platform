import os
import pandas as pd
import numpy as np
import pickle
import warnings
from tqdm import tqdm
from tensorflow.keras.models import load_model
from loguru import logger

# ========================
# CONFIGURACAO
# ========================
RESULTS_DIR = os.path.join('resultados', 'forecasting_deep')
METRICAS_DIR = os.path.join(RESULTS_DIR, 'metricas')
PREVISOES_DIR = os.path.join(RESULTS_DIR, 'previsoes')
MODELOS_DIR = 'modelos_forecasting'
DADOS_DIR = 'dados_transformados'
LOGS_DIR = 'logs'

DEFAULT_TO_CSV_KWARGS = dict(
    sep=';',
    decimal=',',
    index=False
)

os.makedirs(LOGS_DIR, exist_ok=True)
logger.add(os.path.join(LOGS_DIR, 'gerar_ranking_previsoes.log'), level='INFO', rotation='10 MB', encoding='utf-8')

# ========================
# FUNCOES
# ========================

def preparar_dados_forecast(caminho_arquivo):
    df = pd.read_csv(caminho_arquivo, sep=';', decimal=',')
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['Adj Close'])
    return df

def normalizar_serie(serie):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    norm = scaler.fit_transform(serie.values.reshape(-1, 1)).flatten()
    return norm, scaler

def desnormalizar_serie(norm, scaler):
    return scaler.inverse_transform(norm.reshape(-1, 1)).flatten()

# ========================
# EXECUCAO PRINCIPAL
# ========================

if __name__ == "__main__":
    logger.info("Iniciando geração de ranking e previsões dos melhores modelos...")

    df_metricas = pd.read_csv(os.path.join(METRICAS_DIR, 'metricas_forecasting_deep.csv'), sep=';', decimal=',')

    # Ranking: modelo com menor MSE por (arquivo, ticker)
    df_ranking = df_metricas.loc[df_metricas.groupby(['arquivo', 'ticker'])['mse'].idxmin()].copy()

    # Ranking global (ordenado)
    df_ranking = df_ranking.sort_values(by='mse').reset_index(drop=True)
    df_ranking['rank_global'] = df_ranking.index + 1

    # Salvar ranking global
    df_ranking.to_csv(os.path.join(METRICAS_DIR, 'ranking_modelos.csv'), **DEFAULT_TO_CSV_KWARGS)
    logger.info(f"[OK] Ranking global salvo em {os.path.join(METRICAS_DIR, 'ranking_modelos.csv')}")

    # Agora gerar rankings e previsões por arquivo (classe de ativo)
    arquivos_unicos = df_ranking['arquivo'].unique()

    for arquivo in arquivos_unicos:
        logger.info(f"Processando ranking e previsões para arquivo: {arquivo}")

        # Subset do ranking para este arquivo
        df_ranking_arquivo = df_ranking[df_ranking['arquivo'] == arquivo].copy()
        df_ranking_arquivo = df_ranking_arquivo.sort_values(by='mse').reset_index(drop=True)
        df_ranking_arquivo['rank_local'] = df_ranking_arquivo.index + 1

        # Nome base para o arquivo (tirar .csv)
        nome_base = arquivo.replace('.csv', '')

        # Salvar ranking local
        df_ranking_arquivo.to_csv(os.path.join(METRICAS_DIR, f'ranking_modelos_{nome_base}.csv'), **DEFAULT_TO_CSV_KWARGS)
        logger.info(f"[OK] Ranking local salvo em ranking_modelos_{nome_base}.csv")

        # Previsões dos top modelos deste arquivo
        previsoes_top_linhas = []

        for idx, row in tqdm(df_ranking_arquivo.iterrows(), total=len(df_ranking_arquivo), desc=f"Prevendo 15 dias - {nome_base}"):
            try:
                ticker = row['ticker']
                modelo_nome = row['modelo']

                logger.info(f"Prevendo 15 dias: {arquivo} | {ticker} | {modelo_nome}")

                # Recarregar os dados do ticker
                caminho = os.path.join(DADOS_DIR, arquivo)
                df = preparar_dados_forecast(caminho)
                df_ticker = df[df['Ticker'] == ticker].copy()
                df_ticker.sort_values('Date', inplace=True)
                series = df_ticker.set_index('Date')['Adj Close']
                norm_series, scaler = normalizar_serie(series)

                if modelo_nome == 'RandomForest':
                    model_dict = pickle.load(open(os.path.join(MODELOS_DIR, f'{ticker}_rf.pkl'), 'rb'))
                    model_rf = model_dict['modelo']
                    scaler_rf = model_dict['scaler']

                    last_values = norm_series[-10:].tolist()
                    preds_norm = []
                    for _ in range(15):
                        X_input = np.array(last_values[-10:]).reshape(1, -1)
                        pred = model_rf.predict(X_input)[0]
                        preds_norm.append(pred)
                        last_values.append(pred)

                    preds_real = desnormalizar_serie(np.array(preds_norm), scaler_rf)

                elif modelo_nome == 'XGBoost':
                    model_dict = pickle.load(open(os.path.join(MODELOS_DIR, f'{ticker}_xgb.pkl'), 'rb'))
                    model_xgb = model_dict['modelo']
                    scaler_xgb = model_dict['scaler']

                    last_values = norm_series[-10:].tolist()
                    preds_norm = []
                    for _ in range(15):
                        X_input = np.array(last_values[-10:]).reshape(1, -1)
                        pred = model_xgb.predict(X_input)[0]
                        preds_norm.append(pred)
                        last_values.append(pred)

                    preds_real = desnormalizar_serie(np.array(preds_norm), scaler_xgb)

                elif modelo_nome == 'LSTM':
                    model_lstm = load_model(os.path.join(MODELOS_DIR, f'{ticker}_lstm.h5'))
                    scaler_lstm_dict = pickle.load(open(os.path.join(MODELOS_DIR, f'{ticker}_lstm_scaler.pkl'), 'rb'))
                    scaler_lstm = scaler_lstm_dict['scaler']

                    last_values = norm_series[-10:].tolist()
                    preds_norm = []
                    for _ in range(15):
                        X_input = np.array(last_values[-10:]).reshape(1, 10, 1)
                        pred = model_lstm.predict(X_input, verbose=0)[0][0]
                        preds_norm.append(pred)
                        last_values.append(pred)

                    preds_real = desnormalizar_serie(np.array(preds_norm), scaler_lstm)

                else:
                    logger.warning(f"[WARN] Modelo desconhecido: {modelo_nome}. Pulando...")
                    continue

                # Gerar linhas formatadas para Power BI
                for i, valor in enumerate(preds_real):
                    previsoes_top_linhas.append({
                        'arquivo': arquivo,
                        'ticker': ticker,
                        'modelo': modelo_nome,
                        'dia_previsto': i + 1,
                        'valor_previsto': valor
                    })

            except Exception as e_previsao:
                logger.exception(f"[ERROR] Erro ao prever 15 dias para {ticker} em {arquivo}: {str(e_previsao)}")

        # Salvar previsoes top formatadas (1 arquivo por tipo de ativo)
        df_previsoes_top = pd.DataFrame(previsoes_top_linhas)
        df_previsoes_top.to_csv(os.path.join(PREVISOES_DIR, f'previsoes_top_modelos_15dias_{nome_base}.csv'), **DEFAULT_TO_CSV_KWARGS)
        logger.info(f"[OK] Previsões 15 dias salvas em previsoes_top_modelos_15dias_{nome_base}.csv")

    logger.info("Processo finalizado.")
