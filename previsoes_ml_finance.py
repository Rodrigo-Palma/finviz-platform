import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from loguru import logger
import warnings

warnings.filterwarnings('ignore')

# ==========================
# CONFIGURAÇÃO DO LOGGER
# ==========================
os.makedirs('logs', exist_ok=True)
logger.add('logs/predicao_financeira.log', level='INFO', rotation='10 MB', encoding='utf-8')
os.makedirs('previsoes', exist_ok=True)

# ==========================
# FUNÇÕES PRINCIPAIS
# ==========================
def carregar_configuracoes():
    with open('selecionadas/features_selecionadas.json', 'r') as f:
        features_dict = json.load(f)

    janelas_df = pd.read_csv('otimizacao_janela/melhores_janelas.csv')
    janelas_dict = dict(zip(janelas_df['Dataset'], janelas_df['Melhor_Janela']))

    return features_dict, janelas_dict

def converter_janela(janela_str):
    if janela_str.endswith('m'):
        return int(janela_str.replace('m', '')) * 30
    elif janela_str.endswith('a'):
        return int(janela_str.replace('a', '')) * 365
    else:
        return 365  # padrão

def preparar_dados_para_predicao(df, features, dias_retroativos):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Ticker', 'Date'])

    df = df.dropna(subset=['Adj Close'])
    data_limite = df['Date'].max() - timedelta(days=dias_retroativos)
    df = df[df['Date'] >= data_limite]

    df['Target'] = (df.groupby('Ticker')['Adj Close'].shift(-1) > df['Adj Close']).astype(int)

    df = df.dropna(subset=features)
    df[features] = df[features].replace([np.inf, -np.inf], np.nan)
    df[features] = df[features].fillna(df[features].mean())

    if df.empty:
        return pd.DataFrame()

    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    return df

def prever_proximo_dia(arquivo, features_dict, janelas_dict):
    nome_base = os.path.basename(arquivo)
    if nome_base not in features_dict:
        logger.warning(f"{nome_base} não possui features selecionadas. Ignorando...")
        return

    try:
        features = features_dict[nome_base]
        janela = janelas_dict.get(nome_base, '3m')
        dias = converter_janela(janela)

        logger.info(f"{nome_base} | Janela usada: {janela} | Features: {len(features)}")

        df = pd.read_csv(arquivo)
        df_proc = preparar_dados_para_predicao(df, features, dias)

        if df_proc.empty:
            logger.warning(f"{nome_base}: dados insuficientes para previsão.")
            return

        modelo_path = os.path.join('modelos', nome_base.replace('.csv', '_modelo.pkl'))
        model = joblib.load(modelo_path)

        df_final = df_proc.groupby('Ticker').tail(1).copy()
        X = df_final[features]
        df_final['Probabilidade_Alta'] = model.predict_proba(X)[:, 1]
        df_final['Previsao'] = (df_final['Probabilidade_Alta'] >= 0.5).astype(int)

        df_final = df_final[['Ticker', 'Date', 'Probabilidade_Alta', 'Previsao']]
        df_final.to_csv(f"previsoes/previsao_{nome_base}", index=False)

        logger.info(f"{nome_base} | Previsão salva com {len(df_final)} entradas.")
    except Exception as e:
        logger.exception(f"Erro ao processar {nome_base}: {e}")

# ==========================
# EXECUÇÃO
# ==========================
if __name__ == "__main__":
    logger.info("Iniciando previsões com modelos treinados...")
    features_dict, janelas_dict = carregar_configuracoes()
    arquivos = [os.path.join('dados_transformados', f) for f in os.listdir('dados_transformados') if f.endswith('.csv')]

    for arq in arquivos:
        prever_proximo_dia(arq, features_dict, janelas_dict)

    logger.info("Previsões concluídas com sucesso!")
