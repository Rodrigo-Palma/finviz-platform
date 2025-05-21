import os
import json
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from tqdm import tqdm
from loguru import logger
import warnings

warnings.filterwarnings('ignore')

# ==========================
# CONFIGURAÇÃO DE PASTAS
# ==========================
os.makedirs('logs', exist_ok=True)
os.makedirs('otimizacao_janela', exist_ok=True)
os.makedirs('dados_transformados', exist_ok=True)
os.makedirs('selecionadas', exist_ok=True)

# ==========================
# CONFIGURAÇÃO DO LOGGER
# ==========================
from loguru import logger
logger.add('logs/ml_otimizador_janela.log', level='INFO', rotation='10 MB', encoding='utf-8')

# ==========================
# FUNÇÕES AUXILIARES
# ==========================

def preparar_dados(caminho_arquivo, features):
    df = pd.read_csv(caminho_arquivo)
    df['Date'] = pd.to_datetime(df['Date'], utc=False)
    df = df.dropna(subset=features + ['Adj Close'])
    df['Target'] = (df.groupby('Ticker')['Adj Close'].shift(-1) > df['Adj Close']).astype(int)
    return df

def treinar_avaliar(df, features, meses_retroativos, min_amostras=100):
    data_limite = df['Date'].max() - timedelta(days=30*meses_retroativos)
    df_recente = df[df['Date'] >= data_limite]

    if len(df_recente) < min_amostras:
        return None, None, None

    X = df_recente[features]
    y = df_recente['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc = roc_auc_score(y_test, probs)

    return acc, f1, roc

def formatar_nome_janela(meses):
    if meses >= 12:
        return f"{meses//12}a"
    else:
        return f"{meses}m"

# ==========================
# EXECUÇÃO PRINCIPAL
# ==========================

if __name__ == "__main__":
    logger.info("Iniciando otimizacao de janela temporal...")

    janelas_meses = [3, 6, 9, 12, 24, 36]

    arquivos = [f for f in os.listdir('dados_transformados') if f.endswith('.csv')]
    resultados = []

    # Carrega features selecionadas
    with open('selecionadas/features_selecionadas.json', 'r') as f:
        features_especificas = json.load(f)

    for arquivo in tqdm(arquivos, desc="Otimizando janelas"):
        try:
            caminho = os.path.join('dados_transformados', arquivo)
            if arquivo not in features_especificas:
                logger.warning(f"Sem features para {arquivo}, pulando...")
                continue

            df = preparar_dados(caminho, features_especificas[arquivo])
            melhor_roc = 0
            melhor_acc = 0
            melhor_f1 = 0
            melhor_janela = None

            for meses in janelas_meses:
                acc, f1, roc = treinar_avaliar(df, features_especificas[arquivo], meses)
                if roc is not None and roc > melhor_roc:
                    melhor_roc = roc
                    melhor_acc = acc
                    melhor_f1 = f1
                    melhor_janela = formatar_nome_janela(meses)

            resultados.append({
                'Dataset': arquivo,
                'Melhor_Janela': melhor_janela,
                'Melhor_ROC_AUC': melhor_roc,
                'Melhor_Accuracy': melhor_acc,
                'Melhor_F1_Score': melhor_f1
            })

            logger.info(f"{arquivo} - Melhor janela: {melhor_janela} | ROC-AUC={melhor_roc:.4f} | Acc={melhor_acc:.4f} | F1={melhor_f1:.4f}")

        except Exception as e:
            logger.exception(f"Erro processando {arquivo}: {e}")

    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv('otimizacao_janela/melhores_janelas.csv', index=False)
    logger.info("Otimizacao concluida! Resultados salvos em otimizacao_janela/melhores_janelas.csv")
