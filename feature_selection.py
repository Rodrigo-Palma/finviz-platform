import os
import pandas as pd
import numpy as np
import json
import shap
import matplotlib.pyplot as plt
import warnings
import optuna
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ========== CONFIGURAÇÃO ==========
os.makedirs('logs', exist_ok=True)
logger.add("logs/feature_selection_optuna.log", level="INFO", rotation="10 MB", encoding="utf-8")
os.makedirs('selecionadas', exist_ok=True)
os.makedirs('shap_plots', exist_ok=True)
SEED = 42
MIN_LINHAS = 100

# ========== FUNÇÕES ==========
def carregar_dados(caminho_arquivo, limiar_preenchimento=0.1):
    df = pd.read_csv(caminho_arquivo)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')

    df = df.dropna(subset=['Adj Close'])
    df['Target'] = (df.groupby('Ticker')['Adj Close'].shift(-1) > df['Adj Close']).astype(int)

    colunas_excluidas = ['Date', 'Ticker', 'Target', 'Close', 'Excess_Return']
    colunas_validas = [
        col for col in df.columns
        if col not in colunas_excluidas
        and pd.api.types.is_numeric_dtype(df[col])
        and (df[col].notna().mean() >= limiar_preenchimento)
    ]

    removidas = set(df.columns) - set(colunas_validas) - set(colunas_excluidas)
    if removidas:
        logger.warning(f"Removendo colunas com preenchimento insuficiente: {list(removidas)}")
    logger.info(f"{os.path.basename(caminho_arquivo)}: {len(colunas_validas)} features válidas após limpeza.")

    df[colunas_validas] = df[colunas_validas].replace([np.inf, -np.inf], np.nan)
    df[colunas_validas] = df[colunas_validas].fillna(df[colunas_validas].mean())

    df = df[np.isfinite(df[colunas_validas]).all(axis=1)]
    if df.empty or len(df) < MIN_LINHAS:
        logger.warning(f"Arquivo sem dados válidos após limpeza ou muito pequeno: {caminho_arquivo}")
        return pd.DataFrame(), []

    scaler = StandardScaler()
    df[colunas_validas] = scaler.fit_transform(df[colunas_validas])

    return df, colunas_validas

def avaliar_features(df, features, top_k, seed=SEED):
    X = df[features]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2, random_state=seed)
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", verbosity=0, random_state=seed)
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    shap_df = pd.DataFrame(np.abs(shap_values), columns=X.columns).mean().sort_values(ascending=False)
    top_features = shap_df.head(top_k).index.tolist()

    X_train_top = X_train[top_features]
    X_test_top = X_test[top_features]
    model.fit(X_train_top, y_train)
    y_pred = model.predict(X_test_top)

    auc = roc_auc_score(y_test, model.predict_proba(X_test_top)[:, 1])
    acc = accuracy_score(y_test, y_pred)

    return auc, acc, top_features, shap_values, top_k

def selecionar_features_com_optuna(df, features, n_trials=25, nome_dataset=None):
    trial_metrics = []
    def objective(trial):
        top_k = trial.suggest_int("top_k", 10, min(30, len(features)))
        auc, acc, _, _, _ = avaliar_features(df, features, top_k, seed=SEED)
        trial_metrics.append({'top_k': top_k, 'auc': auc, 'acc': acc})
        return auc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    best_k = study.best_params["top_k"]
    logger.info(f"Melhor número de features: {best_k}")
    auc, acc, top_features, shap_values, _ = avaliar_features(df, features, best_k, seed=SEED)

    shap.summary_plot(shap_values, df[features], plot_type="bar", show=False)
    plt.tight_layout()
    shap_name = f"shap_plots/shap_summary_{nome_dataset}.png" if nome_dataset else f"shap_plots/shap_summary_{np.random.randint(10000)}.png"
    plt.savefig(shap_name)
    plt.close()

    logger.info(f"Acurácia: {acc:.4f} | AUC: {auc:.4f}")
    # Salva métricas dos trials
    if nome_dataset:
        pd.DataFrame(trial_metrics).to_csv(f'selecionadas/optuna_trials_{nome_dataset}.csv', index=False)
    return top_features

# ========== EXECUÇÃO ==========
if __name__ == "__main__":
    logger.info("Iniciando seleção de features com Optuna + SHAP...")

    arquivos = [f for f in os.listdir("dados_transformados") if f.endswith(".csv")]
    features_totais = {}
    
    for arquivo in tqdm(arquivos, desc="Selecionando features"):
        try:
            caminho = os.path.join("dados_transformados", arquivo)
            df, features = carregar_dados(caminho)

            if df.empty or not features:
                logger.warning(f"Arquivo ignorado (sem dados válidos): {arquivo}")
                continue

            features_importantes = selecionar_features_com_optuna(df, features, n_trials=10, nome_dataset=arquivo.replace('.csv',''))
            logger.info(f"Features selecionadas para {arquivo}: {features_importantes}")
            features_totais[arquivo] = features_importantes

        except Exception as e:
            logger.exception(f"Erro no processamento de {arquivo}: {e}")

    with open("selecionadas/features_selecionadas.json", "w") as f:
        json.dump(features_totais, f, indent=4)

    logger.info("Seleção de features concluída com Optuna + SHAP.")
