# predict_pipeline_finance.py (corrigido para usar features corretas da seleção)

import os
import json
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.preprocessing import StandardScaler
import pickle
import warnings

warnings.filterwarnings('ignore')

# ==========================
# CONFIGURAÇÃO DO LOGGER
# ==========================
os.makedirs('logs', exist_ok=True)
logger.add('logs/predict_pipeline_finance.log', level='INFO', rotation='10 MB', encoding='utf-8')

# ==========================
# FUNÇÕES DE PIPELINE
# ==========================

def carregar_modelo(caminho_modelo):
    with open(caminho_modelo, 'rb') as file:
        model = pickle.load(file)
    return model

def preparar_dados_predicao(caminho_arquivo, features_desejadas):
    df = pd.read_csv(caminho_arquivo)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])

    df = df.dropna(subset=['Adj Close'])

    features = [feat for feat in features_desejadas if feat in df.columns]

    if not features:
        raise ValueError("Nenhuma feature válida encontrada no dataset.")

    df = df.dropna(subset=features)

    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    return df, features

def gerar_predicoes(modelo, df, features):
    X = df[features]
    preds = modelo.predict(X)
    probs = modelo.predict_proba(X)[:, 1]

    df['Predicted_Target'] = preds
    df['Probability'] = probs

    return df[['Date', 'Ticker', 'Adj Close', 'Predicted_Target', 'Probability']]

# ==========================
# EXECUÇÃO PRINCIPAL
# ==========================

if __name__ == "__main__":
    logger.info("Iniciando pipeline de predições...")

    pasta_modelos = 'modelos'
    pasta_dados = 'dados_transformados'
    pasta_predicoes = 'predicoes'

    os.makedirs(pasta_predicoes, exist_ok=True)

    # Carrega as features selecionadas corretamente
    try:
        with open('selecionadas/features_selecionadas.json', 'r') as f:
            features_especificas = json.load(f)
    except Exception as e:
        logger.error(f"Erro ao carregar features selecionadas: {e}")
        exit()

    modelos = [f for f in os.listdir(pasta_modelos) if f.endswith('_modelo.pkl')]

    for modelo_nome in modelos:
        try:
            logger.info(f"Processando modelo: {modelo_nome}")

            nome_arquivo = modelo_nome.replace('_modelo.pkl', '.csv')
            caminho_modelo = os.path.join(pasta_modelos, modelo_nome)
            caminho_dados = os.path.join(pasta_dados, nome_arquivo)

            if not os.path.exists(caminho_dados):
                logger.warning(f"Arquivo de dados não encontrado para {modelo_nome}")
                continue

            if nome_arquivo not in features_especificas:
                logger.warning(f"Features não encontradas para {nome_arquivo}")
                continue

            modelo = carregar_modelo(caminho_modelo)
            df, features = preparar_dados_predicao(caminho_dados, features_especificas[nome_arquivo])

            if df.empty:
                logger.warning(f"Sem dados suficientes para {modelo_nome}")
                continue

            df_pred = gerar_predicoes(modelo, df, features)

            caminho_saida = os.path.join(pasta_predicoes, nome_arquivo.replace('.csv', '_predicoes.csv'))
            df_pred.to_csv(caminho_saida, index=False)
            logger.info(f"Predições salvas em: {caminho_saida}")

        except Exception as e:
            logger.exception(f"Erro processando {modelo_nome}: {e}")

    logger.info("Pipeline de predições concluído!")
