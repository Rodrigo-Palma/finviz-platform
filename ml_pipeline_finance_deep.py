import os
import json
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from datetime import timedelta
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger

# =============================
# CONFIGURAÇÃO DO LOGGER
# =============================
os.makedirs('logs', exist_ok=True)
logger.add('logs/ml_pipeline_finance_deep.log', level='INFO', rotation='10 MB', encoding='utf-8')

# =============================
# CONFIGURAÇÃO GLOBAL
# =============================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Dispositivo de treino: {DEVICE}")

# =============================
# FUNÇÕES AUXILIARES
# =============================

def garantir_pastas():
    os.makedirs('dados_transformados', exist_ok=True)
    os.makedirs('modelos', exist_ok=True)
    os.makedirs('curvas_treino', exist_ok=True)

def preparar_dados(caminho_arquivo, features_desejadas, janela_anos=None):
    df = pd.read_csv(caminho_arquivo)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])

    df = df.dropna(subset=['Adj Close'])

    if janela_anos is not None:
        # Ajuste janela
        if isinstance(janela_anos, str):
            if janela_anos.endswith('m'):
                meses = int(janela_anos.replace('m', ''))
                dias = meses * 30  # Aproximação: 1 mês ≈ 30 dias
            elif janela_anos.endswith('a'):
                anos = int(janela_anos.replace('a', ''))
                dias = anos * 365
            else:
                raise ValueError(f"Formato de janela inválido: {janela_anos}")
        elif isinstance(janela_anos, (int, float)):
            dias = int(janela_anos * 365)
        else:
            dias = 365 * 10  # Se None, usa 10 anos como fallback

        data_limite = df['Date'].max() - timedelta(days=dias)

        df = df[df['Date'] >= data_limite]
        logger.info(f"Usando janela de {janela_anos} anos para treinamento")

    df['Target'] = (df.groupby('Ticker')['Adj Close'].shift(-1) > df['Adj Close']).astype(int)

    features = [feat for feat in features_desejadas if feat in df.columns]
    df = df.dropna(subset=features)

    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    return df, features

class RedeNeural(nn.Module):
    def __init__(self, input_dim, hidden_layers, activation_fn):
        super().__init__()
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(activation_fn())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# =============================
# FUNÇÃO DE TREINAMENTO (COM EARLY STOPPING)
# =============================

def treinar_modelo_dl(model, optimizer, criterion, train_loader, val_loader, num_epochs=100, patience=10):
    best_val_loss = float('inf')
    patience_counter = 0
    losses = []
    accuracies = []
    aucs = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        preds = []
        targets = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                preds.extend(outputs.cpu().numpy())
                targets.extend(y_batch.cpu().numpy())

        preds = np.array(preds) > 0.5
        acc = accuracy_score(targets, preds)
        auc = roc_auc_score(targets, preds)

        losses.append(val_loss / len(val_loader))
        accuracies.append(acc)
        aucs.append(auc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping na época {epoch+1}")
            break

    return model, losses, accuracies, aucs

# =============================
# OTIMIZAÇÃO OPTUNA
# =============================

def objetivo_optuna(trial, X_train, y_train, X_val, y_val):
    num_layers = trial.suggest_int('num_layers', 1, 3)
    hidden_layers = [trial.suggest_int(f'n_units_l{i}', 32, 256) for i in range(num_layers)]
    activation_name = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU'])
    activation_fn = getattr(nn, activation_name)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    model = RedeNeural(X_train.shape[1], hidden_layers, activation_fn).to(DEVICE)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    train_dataset = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1))
    val_dataset = TensorDataset(torch.tensor(X_val.values, dtype=torch.float32), torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model, _, accuracies, _ = treinar_modelo_dl(model, optimizer, criterion, train_loader, val_loader, num_epochs=50, patience=5)

    return np.mean(accuracies)

# =============================
# SALVAR CURVA
# =============================

def salvar_curva(nome, losses, accuracies, aucs):
    fig, ax1 = plt.subplots()
    ax1.plot(losses, label='Val Loss')
    ax1.set_ylabel('Loss')
    ax2 = ax1.twinx()
    ax2.plot(accuracies, label='Accuracy', color='green')
    ax2.plot(aucs, label='AUC', color='red')
    ax2.set_ylabel('Accuracy / AUC')
    fig.legend(loc='upper right')
    plt.title('Training Metrics')
    plt.savefig(f'curvas_treino/{nome}_curve.png')
    plt.close()

# =============================
# EXECUÇÃO PRINCIPAL
# =============================

if __name__ == "__main__":
    garantir_pastas()
    logger.info("Iniciando pipeline de Deep Learning atualizado para finanças...")

    with open('selecionadas/features_selecionadas.json', 'r') as f:
        features_especificas = json.load(f)

    janelas_otimas = pd.read_csv('otimizacao_janela/melhores_janelas.csv')
    janelas_dict = dict(zip(janelas_otimas['Dataset'], janelas_otimas['Melhor_Janela']))

    arquivos = [f for f in os.listdir('dados_transformados') if f.endswith('.csv')]

    for arquivo in tqdm(arquivos, desc="Treinando redes neurais"):
        try:
            caminho_arquivo = os.path.join('dados_transformados', arquivo)

            if arquivo not in features_especificas:
                logger.warning(f"Arquivo {arquivo} sem features selecionadas. Ignorando...")
                continue

            janela_anos = janelas_dict.get(arquivo, None)
            df, features = preparar_dados(caminho_arquivo, features_especificas[arquivo], janela_anos)

            if df.empty or len(features) == 0:
                logger.warning(f"Arquivo ignorado (sem dados suficientes): {arquivo}")
                continue

            X = df[features]
            y = df['Target']

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: objetivo_optuna(trial, X_train, y_train, X_val, y_val), n_trials=50)

            best_params = study.best_trial.params
            logger.info(f"Melhores parametros: {best_params}")

            model_final = RedeNeural(X.shape[1], [best_params[f'n_units_l{i}'] for i in range(best_params['num_layers'])], getattr(nn, best_params['activation'])).to(DEVICE)
            optimizer_final = getattr(optim, best_params['optimizer'])(model_final.parameters(), lr=best_params['lr'])
            criterion_final = nn.BCELoss()

            train_dataset = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1))
            val_dataset = TensorDataset(torch.tensor(X_val.values, dtype=torch.float32), torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1))

            train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'])

            model_final, losses, accuracies, aucs = treinar_modelo_dl(model_final, optimizer_final, criterion_final, train_loader, val_loader, num_epochs=100, patience=10)

            torch.save(model_final.state_dict(), os.path.join('modelos', arquivo.replace('.csv', '_deep_model.pt')))
            salvar_curva(arquivo.replace('.csv', ''), losses, accuracies, aucs)

            logger.info(f"Modelo treinado e salvo: {arquivo}")

        except Exception as e:
            logger.exception(f"Erro processando {arquivo}: {e}")

    logger.info("Pipeline de Deep Learning atualizado finalizado!")
