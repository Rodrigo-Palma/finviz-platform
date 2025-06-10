import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from loguru import logger
import ast

# ========================
# CONFIGURACAO
# ========================
RESULTS_DIR = os.path.join('resultados', 'forecasting_markov')
LOGS_DIR = 'logs'

DEFAULT_TO_CSV_KWARGS = dict(
    sep=';',
    decimal=',',
    index=False
)

os.makedirs(LOGS_DIR, exist_ok=True)
logger.add(os.path.join(LOGS_DIR, 'gerar_ranking_previsoes_markov.log'), level='INFO', rotation='10 MB', encoding='utf-8')

# ========================
# FUNÇÃO AUXILIAR SEGURA
# ========================
def parse_previsto_safe(valor_str):
    try:
        lista = ast.literal_eval(valor_str)
        if isinstance(lista, list) and len(lista) > 0:
            return lista
        else:
            return None
    except:
        return None

# ========================
# EXECUCAO PRINCIPAL
# ========================
if __name__ == "__main__":
    logger.info("Iniciando geração de ranking e previsões dos melhores modelos (Markov)...")

    df_resultados = pd.read_csv(os.path.join(RESULTS_DIR, 'resultados_forecasting.csv'), sep=';', decimal=',')

    # Converter coluna metricas_modelo de str para dict
    df_resultados['metricas_modelo'] = df_resultados['metricas_modelo'].apply(ast.literal_eval)

    # Extrair métricas em colunas separadas
    df_resultados['mse'] = df_resultados['metricas_modelo'].apply(lambda x: x.get('rmse', np.nan) ** 2 if x else np.nan)
    df_resultados['mae'] = df_resultados['metricas_modelo'].apply(lambda x: x.get('mae', np.nan) if x else np.nan)
    df_resultados['score'] = df_resultados['metricas_modelo'].apply(lambda x: x.get('score', np.nan) if x else np.nan)
    df_resultados['aic'] = df_resultados['metricas_modelo'].apply(lambda x: x.get('aic', np.nan) if x else np.nan)
    df_resultados['bic'] = df_resultados['metricas_modelo'].apply(lambda x: x.get('bic', np.nan) if x else np.nan)
    df_resultados['n_components'] = df_resultados['metricas_modelo'].apply(lambda x: x.get('n_components', np.nan) if x else np.nan)

    # Ranking global (menor MSE)
    df_ranking = df_resultados.sort_values(by='mse').reset_index(drop=True)
    df_ranking['rank_global'] = df_ranking.index + 1

    df_ranking.to_csv(os.path.join(RESULTS_DIR, 'ranking_modelos.csv'), **DEFAULT_TO_CSV_KWARGS)
    logger.info(f"[OK] Ranking global salvo em {os.path.join(RESULTS_DIR, 'ranking_modelos.csv')}")

    # Rankings e previsoes por arquivo
    arquivos_unicos = df_ranking['arquivo'].unique()

    for arquivo in arquivos_unicos:
        logger.info(f"Processando ranking e previsões para arquivo: {arquivo}")

        df_ranking_arquivo = df_ranking[df_ranking['arquivo'] == arquivo].copy()
        df_ranking_arquivo = df_ranking_arquivo.sort_values(by='mse').reset_index(drop=True)
        df_ranking_arquivo['rank_local'] = df_ranking_arquivo.index + 1

        nome_base = arquivo.replace('.csv', '')

        # Salvar ranking local
        df_ranking_arquivo.to_csv(os.path.join(RESULTS_DIR, f'ranking_modelos_{nome_base}.csv'), **DEFAULT_TO_CSV_KWARGS)
        logger.info(f"[OK] Ranking local salvo em ranking_modelos_{nome_base}.csv")

        # Previsões dos top modelos deste arquivo (Power BI friendly)
        previsoes_top_linhas = []

        for idx, row in tqdm(df_ranking_arquivo.iterrows(), total=len(df_ranking_arquivo), desc=f"Prevendo 15 dias - {nome_base}"):
            try:
                ticker = row['ticker']
                regime = row['regime']
                volatilidade = row['volatilidade']

                # Parse seguro da lista de previsões
                previsoes = parse_previsto_safe(row['previsto'])

                if previsoes is None:
                    logger.warning(f"[SKIP] Ignorando {ticker} em {arquivo} pois não possui previsões válidas.")
                    continue

                previsoes = np.array(previsoes)
                n_previsoes = min(len(previsoes), 15)
                previsoes_15 = previsoes[-n_previsoes:]

                for i, valor in enumerate(previsoes_15):
                    previsoes_top_linhas.append({
                        'arquivo': arquivo,
                        'ticker': ticker,
                        'regime': regime,
                        'volatilidade': volatilidade,
                        'modelo': f'HMM_{int(row["n_components"])}_estados',
                        'dia_previsto': i + 1,
                        'valor_previsto': valor
                    })

            except Exception as e_previsao:
                logger.exception(f"[ERROR] Erro ao gerar previsoes para {ticker} em {arquivo}: {e_previsao}")

        # Salvar previsoes top formatadas
        df_previsoes_top = pd.DataFrame(previsoes_top_linhas)
        df_previsoes_top.to_csv(os.path.join(RESULTS_DIR, f'previsoes_top_modelos_15dias_{nome_base}.csv'), **DEFAULT_TO_CSV_KWARGS)
        logger.info(f"[OK] Previsões 15 dias salvas em previsoes_top_modelos_15dias_{nome_base}.csv")

    logger.info("Processo finalizado.")
