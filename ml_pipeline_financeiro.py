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
from xgboost import XGBClassifier
import warnings
from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings('ignore')

# ========================
# CONFIGURACAO GLOBAL
# ========================
RESULTS_DIR = os.path.join('resultados', 'financeiro')
AVALIACAO_DIR = os.path.join(RESULTS_DIR, 'avaliacao')
MODELOS_DIR = os.path.join(RESULTS_DIR, 'modelos')
TRIALS_DIR = os.path.join(RESULTS_DIR, 'trials')
LOGS_DIR = 'logs'
DADOS_DIR = 'dados_transformados'

DEFAULT_TO_CSV_KWARGS = dict(
    sep=';',
    decimal=',',
    index=False
)

for d in [RESULTS_DIR, AVALIACAO_DIR, MODELOS_DIR, TRIALS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

logger.add(os.path.join(LOGS_DIR, 'ml_pipeline_financeiro.log'), level='INFO', rotation='10 MB', encoding='utf-8')

# ========================
# FUNCOES DE PIPELINE
# ========================

def preparar_dados(caminho_arquivo, features_desejadas, janela_anos=None):
    df = pd.read_csv(caminho_arquivo, sep=';', decimal=',')
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='mixed')
    df = df.dropna(subset=['Adj Close'])

    if janela_anos is not None:
        dias = 30 if janela_anos == '1m' else 90 if janela_anos == '3m' else 180 if janela_anos == '6m' else \
               365 if janela_anos == '1a' else 365*3 if janela_anos == '3a' else 365*5 if janela_anos == '5a' else 365*10
        df = df[df['Date'] >= df['Date'].max() - timedelta(days=dias)]
        logger.info(f"Janela usada: {janela_anos} ({dias} dias)")

    df['Target'] = (df.groupby('Ticker')['Adj Close'].shift(-1) > df['Adj Close']).astype(int)

    features = [f for f in features_desejadas if f in df.columns]
    missing = set(features_desejadas) - set(features)
    if missing:
        logger.warning(f"Features não encontradas: {missing}")

    df = df.dropna(subset=features)

    inf_mask = np.isinf(df[features].values)
    if inf_mask.any():
        logger.warning(f"Encontrados {inf_mask.sum()} valores infinitos. Substituindo por NaN.")
        df[features] = df[features].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=features)

    X = df[features]
    y = df['Target']

    selector = VarianceThreshold(threshold=0.01)
    X_selected = selector.fit_transform(X)
    selected_features = [f for f, s in zip(features, selector.get_support()) if s]

    corr_matrix = X[selected_features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    selected_features = [f for f in selected_features if f not in to_drop]

    logger.info(f"Features selecionadas: {len(selected_features)} de {len(features)}")

    df[selected_features] = StandardScaler().fit_transform(df[selected_features])
    return df, selected_features

def objetivo_optuna(trial, X, y):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 5),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.03),
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'subsample': trial.suggest_float('subsample', 0.5, 0.7),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.7),
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 15),
        'gamma': trial.suggest_float('gamma', 0.2, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 3.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 3.0),
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }
    model = XGBClassifier(**params)

    n_splits = 5
    scores = []
    for i in range(n_splits):
        split_point = int(len(X) * (i + 1) / (n_splits + 1))
        X_train, X_val = X.iloc[:split_point], X.iloc[split_point:split_point + int(len(X) * 0.2)]
        y_train, y_val = y.iloc[:split_point], y.iloc[split_point:split_point + int(len(X) * 0.2)]

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        preds = model.predict(X_val)
        score = roc_auc_score(y_val, preds)
        logger.debug(f"Walk-forward {i+1} AUC: {score:.4f}")
        scores.append(score)

    return np.mean(scores)

def treinar_modelo(df, features, modelo='xgb'):
    X = df[features]
    y = df['Target']

    N_TRIALS = min(50, max(10, len(df) // 100))
    logger.info(f"Usando {N_TRIALS} trials para otimização")

    models = []
    weights = []

    for i in range(5):
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objetivo_optuna(trial, X, y), n_trials=N_TRIALS)

        best_params = study.best_trial.params
        logger.info(f"Melhores parâmetros para xgb (modelo {i+1}): {best_params}")
        study.trials_dataframe().to_csv(
            os.path.join(TRIALS_DIR, f'trials_xgb_{i+1}.csv'),
            **DEFAULT_TO_CSV_KWARGS
        )

        best_params.update({
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        })
        best_model = XGBClassifier(**best_params)

        best_model.fit(X, y)
        preds = best_model.predict_proba(X)[:, 1]
        weight = roc_auc_score(y, preds)

        models.append(best_model)
        weights.append(weight)

    weights = np.array(weights) / sum(weights)
    logger.info(f"Pesos dos modelos: {weights}")

    return models, weights, features

def avaliar_modelo(models, weights, df, features_final, nome_modelo):
    X = df[features_final]
    y = df['Target']

    n_splits = 5
    cv_scores = []

    for i in range(n_splits):
        split_point = int(len(X) * (i + 1) / (n_splits + 1))
        X_train, X_val = X.iloc[:split_point], X.iloc[split_point:split_point + int(len(X) * 0.2)]
        y_train, y_val = y.iloc[:split_point], y.iloc[split_point:split_point + int(len(X) * 0.2)]

        for model in models:
            model.fit(X_train, y_train)

        preds_proba = np.zeros(len(X_val))
        for model, weight in zip(models, weights):
            preds_proba += weight * model.predict_proba(X_val)[:, 1]

        preds = (preds_proba > 0.5).astype(int)

        acc = accuracy_score(y_val, preds)
        roc = roc_auc_score(y_val, preds_proba)
        cv_scores.append((acc, roc))
        logger.info(f"Walk-forward {i+1} - Acurácia: {acc:.4f} | AUC: {roc:.4f}")

    mean_acc = np.mean([s[0] for s in cv_scores])
    mean_roc = np.mean([s[1] for s in cv_scores])
    std_acc = np.std([s[0] for s in cv_scores])
    std_roc = np.std([s[1] for s in cv_scores])
    logger.info(f"Média Walk-forward - Acurácia: {mean_acc:.4f} (±{std_acc:.4f}) | AUC: {mean_roc:.4f} (±{std_roc:.4f})")

    preds_proba = np.zeros(len(X))
    for model, weight in zip(models, weights):
        preds_proba += weight * model.predict_proba(X)[:, 1]

    preds = (preds_proba > 0.5).astype(int)
    acc = accuracy_score(y, preds)
    roc = roc_auc_score(y, preds_proba)
    report = classification_report(y, preds, output_dict=True)
    logger.info(f"Avaliação Final - Acurácia: {acc:.4f} | AUC: {roc:.4f}")

    overfit_acc = acc - mean_acc
    overfit_roc = roc - mean_roc
    logger.info(f"Indicador de Overfitting - Acurácia: {overfit_acc:.4f} | AUC: {overfit_roc:.4f}")

    df_result = pd.DataFrame(report).T
    df_result['index'] = df_result.index
    df_result.loc['summary'] = [acc, roc, None, None, 'summary']
    df_result.loc['cv_mean'] = [mean_acc, mean_roc, None, None, 'cv_mean']
    df_result.loc['cv_std'] = [std_acc, std_roc, None, None, 'cv_std']
    df_result.loc['overfit'] = [overfit_acc, overfit_roc, None, None, 'overfit']

    df_result.to_csv(
        os.path.join(AVALIACAO_DIR, f'avaliacao_{nome_modelo}.csv'),
        **DEFAULT_TO_CSV_KWARGS
    )

# ========================
# EXECUCAO PRINCIPAL
# ========================

if __name__ == "__main__":
    logger.info("Iniciando pipeline para finanças...")

    with open('selecionadas/features_selecionadas.json', 'r') as f:
        features_especificas = json.load(f)

    janelas_df = pd.read_csv('otimizacao_janela/melhores_janelas.csv', sep=';', decimal=',')
    janelas_dict = dict(zip(janelas_df['Dataset'], janelas_df['Melhor_Janela']))

    arquivos = [f for f in os.listdir(DADOS_DIR) if f.endswith('.csv')]

    for arquivo in tqdm(arquivos, desc="Modelos treinados"):
        try:
            if arquivo not in features_especificas:
                logger.warning(f"{arquivo} não possui features selecionadas.")
                continue

            janela = janelas_dict.get(arquivo, None)
            caminho = os.path.join(DADOS_DIR, arquivo)
            df, selected_features = preparar_dados(caminho, features_especificas[arquivo], janela)

            if df is None or len(selected_features) == 0:
                logger.warning(f"{arquivo} ignorado (sem dados ou features).")
                continue

            inicio = time.time()
            models, weights, selected_features = treinar_modelo(df, selected_features, modelo='xgb')
            avaliar_modelo(models, weights, df, selected_features, nome_modelo=arquivo.replace('.csv', ''))
            fim = time.time()

            nome_modelo = os.path.join(MODELOS_DIR, arquivo.replace('.csv', '_modelo.pkl'))
            joblib.dump((models, weights), nome_modelo)
            logger.info(f"Modelo salvo: {nome_modelo} | Tempo total: {fim - inicio:.2f}s")

        except Exception as e:
            logger.exception(f"Erro ao processar {arquivo}: {e}")

    logger.info("Pipeline concluído com sucesso.")
