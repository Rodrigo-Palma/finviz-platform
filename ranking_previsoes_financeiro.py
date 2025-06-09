import os
import json
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from loguru import logger

# ========================
# CONFIGURACAO
# ========================
RESULTS_DIR = os.path.join('resultados', 'financeiro')
AVALIACAO_DIR = os.path.join(RESULTS_DIR, 'avaliacao')
MODELOS_DIR = os.path.join(RESULTS_DIR, 'modelos')
DADOS_DIR = 'dados_transformados'
LOGS_DIR = 'logs'

DEFAULT_TO_CSV_KWARGS = dict(
    sep=';',
    decimal=',',
    index=False
)

os.makedirs(LOGS_DIR, exist_ok=True)
logger.add(os.path.join(LOGS_DIR, 'gerar_ranking_previsoes_financeiro.log'), level='INFO', rotation='10 MB', encoding='utf-8')

# ========================
# EXECUCAO PRINCIPAL
# ========================

if __name__ == "__main__":
    logger.info("Iniciando geração de ranking e previsões dos melhores modelos (Financeiro)...")

    # Carregar features_selecionadas.json
    try:
        with open('selecionadas/features_selecionadas.json', 'r') as f:
            features_selecionadas = json.load(f)
        logger.info("[OK] Features_selecionadas carregado com sucesso.")
    except Exception as e:
        logger.exception(f"[ERROR] Erro ao carregar features_selecionadas.json: {e}")
        features_selecionadas = {}

    # Carregar todas as avaliações
    avaliacoes_files = [f for f in os.listdir(AVALIACAO_DIR) if f.startswith('avaliacao_') and f.endswith('.csv')]
    logger.info(f"Total de arquivos de avaliação encontrados: {len(avaliacoes_files)}")

    lista_ranking = []

    for file in avaliacoes_files:
        try:
            logger.info(f"Processando avaliação: {file}")
            df_av = pd.read_csv(os.path.join(AVALIACAO_DIR, file), sep=';', decimal=',')

            acc = df_av.loc[df_av['index'] == 'summary', 'precision'].values[0]
            auc = df_av.loc[df_av['index'] == 'summary', 'recall'].values[0]
            overfit_acc = df_av.loc[df_av['index'] == 'overfit', 'precision'].values[0]
            overfit_auc = df_av.loc[df_av['index'] == 'overfit', 'recall'].values[0]

            lista_ranking.append({
                'arquivo': file.replace('avaliacao_', '').replace('.csv', '.csv'),
                'acc': acc,
                'auc': auc,
                'overfit_acc': overfit_acc,
                'overfit_auc': overfit_auc
            })

            logger.info(f"[OK] Avaliação processada com sucesso: {file}")

        except Exception as e:
            logger.exception(f"[ERROR] Erro ao processar avaliação {file}: {e}")

    # Montar DataFrame de ranking
    df_ranking = pd.DataFrame(lista_ranking)
    df_ranking = df_ranking.sort_values(by='auc', ascending=False).reset_index(drop=True)
    df_ranking['rank_global'] = df_ranking.index + 1

    df_ranking.to_csv(os.path.join(RESULTS_DIR, 'ranking_modelos.csv'), **DEFAULT_TO_CSV_KWARGS)
    logger.info(f"[OK] Ranking global salvo em {os.path.join(RESULTS_DIR, 'ranking_modelos.csv')}")

    # Rankings e previsoes por arquivo
    arquivos_unicos = df_ranking['arquivo'].unique()

    for arquivo in arquivos_unicos:
        logger.info(f"Processando ranking e previsões para arquivo: {arquivo}")

        df_ranking_arquivo = df_ranking[df_ranking['arquivo'] == arquivo].copy()
        df_ranking_arquivo = df_ranking_arquivo.sort_values(by='auc', ascending=False).reset_index(drop=True)
        df_ranking_arquivo['rank_local'] = df_ranking_arquivo.index + 1

        nome_base = arquivo.replace('.csv', '')

        df_ranking_arquivo.to_csv(os.path.join(RESULTS_DIR, f'ranking_modelos_{nome_base}.csv'), **DEFAULT_TO_CSV_KWARGS)
        logger.info(f"[OK] Ranking local salvo em ranking_modelos_{nome_base}.csv")

        previsoes_top_linhas = []

        try:
            logger.info(f"Carregando dados transformados para {arquivo}...")
            df = pd.read_csv(os.path.join(DADOS_DIR, arquivo), sep=';', decimal=',', engine='python')
            df['Date'] = pd.to_datetime(df['Date'], format='mixed')
            df = df.dropna(subset=['Adj Close'])
            logger.info(f"[OK] Dados carregados e pré-processados: {len(df)} registros.")

            # Carregar modelo
            modelo_path = os.path.join(MODELOS_DIR, arquivo.replace('.csv', '_modelo.pkl'))
            logger.info(f"Carregando modelo de {modelo_path}...")
            loaded = joblib.load(modelo_path)

            if len(loaded) == 3:
                models, weights, features_modelo = loaded
                logger.info(f"[OK] Modelo carregado com {len(models)} modelos e {len(features_modelo)} features.")
            else:
                models, weights = loaded
                logger.warning(f"[WARN] Modelo {arquivo} não possui features salvas, usando features_selecionadas.json se disponível.")
                features_modelo = features_selecionadas.get(arquivo, [])
                if not features_modelo:
                    logger.warning(f"[WARN] Arquivo {arquivo} não possui features em features_selecionadas.json, usando todas as colunas numéricas.")
                    features_modelo = df.select_dtypes(include=[np.number]).columns.tolist()

            logger.info(f"[OK] Features carregadas de features_selecionadas.json: {len(features_modelo)} features.")

            # Garantir consistência de features
            logger.info(f"Verificando e aplicando as features do modelo...")
            df = df.dropna(subset=features_modelo)

            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            df[features_modelo] = scaler.fit_transform(df[features_modelo])

            # ===== CORREÇÃO PRINCIPAL: últimos 15 registros por Ticker
            if 'Ticker' in df.columns:
                df_ultimos = df.groupby('Ticker', group_keys=False).apply(lambda x: x.tail(15)).reset_index(drop=True)
                logger.info(f"[OK] Últimos 15 registros por Ticker carregados: {df_ultimos['Ticker'].nunique()} tickers, {len(df_ultimos)} registros.")
            else:
                df_ultimos = df[-15:]
                logger.warning(f"[WARN] Coluna 'Ticker' não encontrada, usando últimos 15 registros do DataFrame inteiro.")

            # Booster feature check
            booster_feature_names = models[0].get_booster().feature_names
            logger.info(f"[CHECK] Booster features: {len(booster_feature_names)} features.")

            X_ultimos = df_ultimos[booster_feature_names]

            # ======================== PREDICTIONS
            logger.info(f"Iniciando previsões para os últimos {len(df_ultimos)} registros...")
            preds_proba = np.zeros(len(X_ultimos))
            for i, (model, weight) in enumerate(zip(models, weights)):
                logger.info(f"→ Modelo {i+1}, peso {weight:.4f}")
                preds_proba += weight * model.predict_proba(X_ultimos)[:, 1]

            logger.info("Montando DataFrame de previsões...")

            for i, (idx, row) in enumerate(df_ultimos.iterrows()):
                previsoes_top_linhas.append({
                    'arquivo': arquivo,
                    'ticker': row['Ticker'] if 'Ticker' in df_ultimos.columns else 'N/A',
                    'data': row['Date'].strftime('%Y-%m-%d'),
                    'dia_previsto': i + 1,
                    'probabilidade_subir': preds_proba[i]
                })

            df_previsoes_top = pd.DataFrame(previsoes_top_linhas)
            df_previsoes_top.to_csv(os.path.join(RESULTS_DIR, f'previsoes_top_modelos_15dias_{nome_base}.csv'), **DEFAULT_TO_CSV_KWARGS)
            logger.info(f"[OK] Previsões 15 dias salvas em previsoes_top_modelos_15dias_{nome_base}.csv")

        except Exception as e_previsao:
            logger.exception(f"[ERROR] Erro ao gerar previsoes para {arquivo}: {e_previsao}")

    logger.info("Processo finalizado com sucesso.")
