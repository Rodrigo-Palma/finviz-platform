import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from datetime import timedelta
from loguru import logger


# =============================
# CONFIGURAÇÃO GLOBAL
# =============================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('logs', exist_ok=True)
logger.add('logs/ml_pipeline_finance_deep3.log', level='INFO', rotation='10 MB', encoding='utf-8')
logger.info(f"Dispositivo de treino: {DEVICE}")

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    logger.info(f"Treinando na GPU: {torch.cuda.get_device_name(DEVICE)}")
else:
    DEVICE = torch.device('cpu')
    logger.info("GPU não disponível. Treinando na CPU.")

# =============================
# MODELOS
# =============================
class MLP(nn.Module):
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

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x.unsqueeze(1))  # Add sequence dimension if not present
        h_n = h_n[-1]
        out = self.fc(h_n)
        return torch.sigmoid(out)

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, h_n = self.gru(x.unsqueeze(1))  # Add sequence dimension if not present
        h_n = h_n[-1]
        out = self.fc(h_n)
        return torch.sigmoid(out)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, nhead, num_layers):
        super().__init__()
        # Ensure that embedding dimension is divisible by nhead
        self.embed_dim = nhead * ((input_dim // nhead) + (1 if input_dim % nhead != 0 else 0))
        self.embedding = nn.Linear(input_dim, self.embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(self.embed_dim, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence dimension if not present
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.squeeze(1)  # Remove sequence dimension for fully connected layer
        return torch.sigmoid(self.fc(x))

# =============================
# UTILITÁRIOS
# =============================
def preparar_dados(caminho, features, janela_anos=None):
    df = pd.read_csv(caminho)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna(subset=['Adj Close'])

    if janela_anos:
        dias = 365 * int(janela_anos.replace('a', '')) if 'a' in janela_anos else 30 * int(janela_anos.replace('m', ''))
        df = df[df['Date'] >= (df['Date'].max() - timedelta(days=dias))]

    df['Target'] = (df.groupby('Ticker')['Adj Close'].shift(-1) > df['Adj Close']).astype(int)
    features = [f for f in features if f in df.columns]
    df = df.dropna(subset=features)
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df, features

def treinar(model, optimizer, criterion, train_loader, val_loader, num_epochs=100, patience=10):
    best_loss = float('inf')
    patience_counter = 0
    losses, accs, aucs = [], [], []

    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                preds = model(X_batch)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(y_batch.cpu().numpy())

        val_preds = np.array(val_preds) > 0.5
        acc = accuracy_score(val_targets, val_preds)
        auc = roc_auc_score(val_targets, val_preds)
        losses.append(loss.item())
        accs.append(acc)
        aucs.append(auc)

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    return losses, accs, aucs, np.mean(aucs)

# =============================
# PIPELINE PRINCIPAL
# =============================

if __name__ == "__main__":
    os.makedirs('modelos', exist_ok=True)
    os.makedirs('curvas_treino', exist_ok=True)
    resultados = []

    with open('selecionadas/features_selecionadas.json') as f:
        features_selecionadas = json.load(f)

    arquivos = [f for f in os.listdir('dados_transformados') if f.endswith('.csv')]

    for arquivo in tqdm(arquivos, desc="Treinando arquivos"):
        try:
            df, features = preparar_dados(f'dados_transformados/{arquivo}', features_selecionadas.get(arquivo, []))
            if df.empty:
                continue

            X = df[features].values
            y = df['Target'].values

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
            X_val = torch.tensor(X_val, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

            input_dim = X_train.shape[1]
            possiveis_heads = [h for h in [1, 2, 4, 8] if input_dim % h == 0]
            nhead_transformer = max(possiveis_heads) if possiveis_heads else 1

            modelos = {
                'MLP': MLP(input_dim, [128, 64], nn.ReLU),
                'LSTM': LSTMModel(input_dim, 64, 2),
                'GRU': GRUModel(input_dim, 64, 2),
                'Transformer': TransformerModel(input_dim, nhead=nhead_transformer, num_layers=2)
            }

            melhor_auc = 0
            melhor_modelo_nome = None
            melhor_modelo = None
            melhor_curvas = None

            for nome, model in modelos.items():
                model = model.to(DEVICE)
                optimizer = optim.Adam(model.parameters(), lr=1e-3)
                criterion = nn.BCELoss()
                train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
                val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

                losses, accs, aucs, auc_final = treinar(model, optimizer, criterion, train_loader, val_loader)

                if auc_final > melhor_auc:
                    melhor_auc = auc_final
                    melhor_modelo_nome = nome
                    melhor_modelo = model
                    melhor_curvas = (losses, accs, aucs)

            torch.save(melhor_modelo.state_dict(), f"modelos/{arquivo.replace('.csv', '')}_{melhor_modelo_nome}.pt")
            plt.figure()
            plt.plot(melhor_curvas[0], label='Loss')
            plt.plot(melhor_curvas[1], label='Accuracy')
            plt.plot(melhor_curvas[2], label='AUC')
            plt.legend()
            plt.title(f'{arquivo} - {melhor_modelo_nome} Treinamento')
            plt.savefig(f"curvas_treino/{arquivo.replace('.csv', '')}_{melhor_modelo_nome}_curve.png")
            plt.close()

            resultados.append({
                'Dataset': arquivo,
                'Melhor_Modelo': melhor_modelo_nome,
                'AUC': melhor_auc
            })

            logger.info(f"{arquivo}: Melhor modelo {melhor_modelo_nome} com AUC {melhor_auc:.4f}")

        except Exception as e:
            logger.exception(f"Erro treinando {arquivo}: {e}")

    pd.DataFrame(resultados).to_csv('relatorio_modelos.csv', index=False)
    logger.info("Treinamento concluído!")
