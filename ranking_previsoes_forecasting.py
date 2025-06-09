import os
import pandas as pd
import numpy as np
import pickle
import warnings
from tqdm import tqdm
from loguru import logger

# ========================
# CONFIGURACAO
# ========================
RESULTS_DIR = os.path.join('resultados', 'forecasting_arima_prophet')
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
logger.add(os.path.join(LOGS_DIR, 'gerar_ranking_previsoes_arima_prophet.log'), level='INFO', rotation='10 MB', encoding='utf-8')

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

# ========================
# EXECUCAO PRINCIPAL
# ========================

if __name__ == "__main__":
    logger.info("Iniciando geração de ranking e previsões dos melhores modelos (ARIMA + Prophet)...")

    df_metricas = pd.read_csv(os.path.join(METRICAS_DIR, 'metricas_forecasting.csv'), sep=';', decimal=',')

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
                real_ultimos_30 = series[-30:].values

                # Carregar modelo salvo
                model_path = None
                min_valor = None
                max_valor = None
                if modelo_nome.startswith('ARIMA'):
                    model_path = os.path.join(MODELOS_DIR, f'{ticker}_arima.pkl')
                elif modelo_nome.startswith('Prophet'):
                    model_path = os.path.join(MODELOS_DIR, f'{ticker}_prophet.pkl')
                else:
                    logger.warning(f"[WARN] Modelo desconhecido: {modelo_nome}. Pulando...")
                    continue

                model_dict = pickle.load(open(model_path, 'rb'))
                modelo_carregado = model_dict['modelo']
                min_valor = model_dict['min']
                max_valor = model_dict['max']

                # Previsão 15 dias
                if modelo_nome.startswith('ARIMA'):
                    forecast_arima = modelo_carregado.forecast(steps=15)
                    preds_real = (forecast_arima.values * (max_valor - min_valor)) + min_valor

                elif modelo_nome.startswith('Prophet'):
                    futuro = modelo_carregado.make_future_dataframe(periods=15)
                    forecast = modelo_carregado.predict(futuro)
                    yhat_15dias = forecast['yhat'][-15:].values
                    preds_real = (yhat_15dias * (max_valor - min_valor)) + min_valor

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
