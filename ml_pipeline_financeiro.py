import os
import json
import optuna
import pandas as pd
import numpy as np
import joblib
import time
from tqdm import tqdm
from datetime import timedelta
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings

warnings.filterwarnings('ignore')

# =======================
# CONFIGURAÇÕES GLOBAIS
# =======================
MODELO_ESCOLHIDO = 'xgb'
N_TRIALS = 50
MIN_REGISTROS = 100

# =======================
# LOGGER E PASTAS
# =======================
os.makedirs('logs', exist_ok=True)
os.makedirs('modelos', exist_ok=True)
os.makedirs('resultados_avaliacao', exist_ok=True)
logger.add('logs/ml_pipeline_finance.log', level='INFO', rotation='10 MB', encoding='utf-8')

# =======================
# FUNÇÕES DE PIPELINE
# =======================

def preparar_dados(caminho_arquivo, features_desejadas, janela_anos=None):
    df = pd.read_csv(caminho_arquivo)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna(subset=['Adj Close'])

    if janela_anos is not None:
        dias = 30 if janela_anos == '1m' else 90 if janela_anos == '3m' else 180 if janela_anos == '6m' else \
               365 if janela_anos == '1a' else 365*3 if janela_anos == '3a' else 365*5 if janela_anos == '5a' else 365*10
        df = df[df['Date'] >= df['Date'].max() - timedelta(days=dias)]
        logger.info(f"Janela usada: {janela_anos} ({dias} dias)")

    df['Target'] = (df.groupby('Ticker')['Adj Close'].shift(-1) > df['Adj Close']).astype(int)
    features = [f for f in features_desejadas if f in df.columns]
    df = df.dropna(subset=features)
    if df.empty or len(df) < MIN_REGISTROS:
        logger.warning(f"Dataset com dados insuficientes após limpeza: {caminho_arquivo}")
        return None, None

    df[features] = StandardScaler().fit_transform(df[features])
    return df, features

def objetivo_optuna(trial, X, y, modelo):
    if modelo == 'xgb':
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        model = XGBClassifier(**params)
    elif modelo == 'lgbm':
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0)
        }
        model = LGBMClassifier(**params)
    elif modelo == 'catboost':
        params = {
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'iterations': trial.suggest_int('iterations', 100, 400),
            'verbose': 0
        }
        model = CatBoostClassifier(**params)
    else:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'max_depth': trial.suggest_int('max_depth', 5, 20)
        }
        model = RandomForestClassifier(**params)

    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        scores.append(roc_auc_score(y_val, preds))
    return np.mean(scores)

def treinar_modelo(df, features, modelo='xgb'):
    X = df[features]
    y = df['Target']

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objetivo_optuna(trial, X, y, modelo), n_trials=N_TRIALS)

    best_params = study.best_trial.params
    logger.info(f"Melhores parâmetros para {modelo}: {best_params}")
    study.trials_dataframe().to_csv('resultados_avaliacao/trials_' + modelo + '.csv', index=False)

    if modelo == 'xgb':
        best_params.update({'use_label_encoder': False, 'eval_metric': 'logloss'})
        best_model = XGBClassifier(**best_params)
    elif modelo == 'lgbm':
        best_model = LGBMClassifier(**best_params)
    elif modelo == 'catboost':
        best_model = CatBoostClassifier(verbose=0, **best_params)
    else:
        best_model = RandomForestClassifier(**best_params)

    best_model.fit(X, y)
    return best_model

def avaliar_modelo(model, df, features, nome_modelo):
    X = df[features]
    y = df['Target']
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    acc = accuracy_score(y, preds)
    roc = roc_auc_score(y, proba)
    report = classification_report(y, preds, output_dict=True)
    logger.info(f"Acurácia: {acc:.4f} | AUC: {roc:.4f}")

    df_result = pd.DataFrame(report).T
    df_result.loc['summary'] = [acc, roc, None, None]
    df_result.to_csv(f'resultados_avaliacao/avaliacao_{nome_modelo}.csv')

# =======================
# EXECUÇÃO PRINCIPAL
# =======================
if __name__ == "__main__":
    logger.info("Iniciando pipeline para finanças...")

    try:
        with open('selecionadas/features_selecionadas.json', 'r') as f:
            features_especificas = json.load(f)
    except Exception as e:
        logger.error(f"Erro ao carregar features selecionadas: {e}")
        exit()

    try:
        janelas_df = pd.read_csv('otimizacao_janela/melhores_janelas.csv')
        janelas_dict = dict(zip(janelas_df['Dataset'], janelas_df['Melhor_Janela']))
    except Exception as e:
        logger.error(f"Erro ao carregar melhores janelas: {e}")
        exit()

    arquivos = [f for f in os.listdir('dados_transformados') if f.endswith('.csv')]

    for arquivo in tqdm(arquivos, desc="Modelos treinados"):
        try:
            if arquivo not in features_especificas:
                logger.warning(f"{arquivo} não possui features selecionadas.")
                continue

            janela = janelas_dict.get(arquivo, None)
            caminho = os.path.join('dados_transformados', arquivo)
            df, features = preparar_dados(caminho, features_especificas[arquivo], janela)

            if df is None or len(features) == 0:
                logger.warning(f"{arquivo} ignorado (sem dados ou features).")
                continue

            inicio = time.time()
            model = treinar_modelo(df, features, modelo=MODELO_ESCOLHIDO)
            avaliar_modelo(model, df, features, nome_modelo=arquivo.replace('.csv', ''))
            fim = time.time()

            nome_modelo = os.path.join('modelos', arquivo.replace('.csv', '_modelo.pkl'))
            joblib.dump(model, nome_modelo)
            logger.info(f"Modelo salvo: {nome_modelo} | Tempo total: {fim - inicio:.2f}s")

        except Exception as e:
            logger.exception(f"Erro ao processar {arquivo}: {e}")

    logger.info("Pipeline concluído com sucesso.")
