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

# Parametrização do número de dias futuros
N_DIAS_FUTURO = 15

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
            df = df.dropna(subset=features_modelo)

            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            df[features_modelo] = scaler.fit_transform(df[features_modelo])

            # Get booster feature names for safety
            booster_feature_names = models[0].get_booster().feature_names
            logger.info(f"[CHECK] Booster features: {len(booster_feature_names)} features.")

            # Loop por ticker
            for ticker, df_ticker in df.groupby('Ticker'):
                logger.info(f"Gerando previsões futuras para Ticker: {ticker}")

                # Pegar última linha do ticker
                last_row = df_ticker.sort_values('Date').iloc[-1]

                # Gerar 15 linhas futuras
                df_future = pd.DataFrame([last_row] * N_DIAS_FUTURO)
                df_future['Date'] = pd.date_range(start=last_row['Date'] + pd.Timedelta(days=1), periods=N_DIAS_FUTURO)

                # Se necessário: recomputar features que dependem da data (placeholder)
                # Exemplo: df_future['feature_X'] = recomputar(df_future['Date'])

                # Garantir as features corretas e ordenadas
                X_future = df_future[booster_feature_names]

                # Prever
                preds_proba = np.zeros(len(X_future))
                for i, (model, weight) in enumerate(zip(models, weights)):
                    logger.info(f"→ Modelo {i+1}, peso {weight:.4f}")
                    preds_proba += weight * model.predict_proba(X_future)[:, 1]

                # Montar resultados
                for i, (idx, row) in enumerate(df_future.iterrows()):
                    previsoes_top_linhas.append({
                        'arquivo': arquivo,
                        'ticker': ticker,
                        'data': row['Date'].strftime('%Y-%m-%d'),
                        'dia_previsto': i + 1,
                        'probabilidade_subir': preds_proba[i]
                    })

            # Salvar previsões
            df_previsoes_top = pd.DataFrame(previsoes_top_linhas)
            df_previsoes_top.to_csv(os.path.join(RESULTS_DIR, f'previsoes_top_modelos_{N_DIAS_FUTURO}dias_{nome_base}.csv'), **DEFAULT_TO_CSV_KWARGS)
            logger.info(f"[OK] Previsões {N_DIAS_FUTURO} dias salvas em previsoes_top_modelos_{N_DIAS_FUTURO}dias_{nome_base}.csv")

        except Exception as e_previsao:
            logger.exception(f"[ERROR] Erro ao gerar previsoes para {arquivo}: {e_previsao}")

    logger.info("Processo finalizado com sucesso.")
