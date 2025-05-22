import os
import pandas as pd
import numpy as np
import warnings
import pickle
from tqdm import tqdm
from loguru import logger
from hmmlearn import hmm
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import Tuple, List, Dict, Optional
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import backend as K

# =====================
# TENTATIVA DE CUPY
# =====================
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    logger.info('CuPy disponível. Backend GPU ativado para operações NumPy.')
except ImportError:
    cp = np  # fallback para NumPy
    CUPY_AVAILABLE = False
    logger.info('CuPy não encontrado. Usando NumPy (CPU) para operações NumPy.')

warnings.filterwarnings("ignore")
np.random.seed(42)

# =====================
# CONFIGURACAO GPU
# =====================
def configurar_gpu():
    """Configura o uso de GPU para TensorFlow e outras operações."""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPU disponível: {gpus}")
            return True
        else:
            logger.warning("Nenhuma GPU encontrada. Usando CPU.")
            return False
    except Exception as e:
        logger.error(f"Erro ao configurar GPU: {e}")
        return False

# Configura GPU
GPU_AVAILABLE = configurar_gpu() and CUPY_AVAILABLE

# =====================
# CONFIGURACAO DE DIRETORIOS
# =====================
RESULTS_DIR = os.path.join('resultados', 'forecasting_markov')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
RESIDUOS_DIR = os.path.join(RESULTS_DIR, 'residuos')
PREVISOES_DIR = os.path.join(RESULTS_DIR, 'previsoes')
METRICAS_DIR = os.path.join(RESULTS_DIR, 'metricas')
ANALISES_DIR = os.path.join(RESULTS_DIR, 'analises')
MODELOS_DIR = os.path.join('modelos_forecasting')
LOGS_DIR = 'logs'

for d in [RESULTS_DIR, PLOTS_DIR, RESIDUOS_DIR, PREVISOES_DIR, METRICAS_DIR, ANALISES_DIR, MODELOS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

logger.add(os.path.join(LOGS_DIR, 'forecast_pipeline_markov.log'), level='INFO', rotation='10 MB', encoding='utf-8')

# =====================
# FUNCOES DE FORECAST
# =====================

class MarkovChainForecaster:
    def __init__(self, n_states: int = 10):
        self.n_states = n_states
        self.transition_matrix = None
        self.state_means = None
        self.state_stds = None
        self.discretizer = KBinsDiscretizer(n_bins=n_states, encode='ordinal', strategy='quantile')
        
    def fit(self, series: np.ndarray) -> None:
        """Treina o modelo de Cadeia de Markov."""
        # Converte para GPU se disponível
        if GPU_AVAILABLE:
            series = cp.asarray(series)
        
        # Discretiza a série em estados
        states = self.discretizer.fit_transform(series.reshape(-1, 1)).flatten()
        
        # Calcula a matriz de transição
        self.transition_matrix = cp.zeros((self.n_states, self.n_states)) if GPU_AVAILABLE else np.zeros((self.n_states, self.n_states))
        for i in range(len(states) - 1):
            self.transition_matrix[int(states[i]), int(states[i + 1])] += 1
        
        # Normaliza a matriz de transição
        row_sums = self.transition_matrix.sum(axis=1)
        self.transition_matrix = cp.divide(self.transition_matrix, row_sums[:, cp.newaxis],
                                         where=row_sums[:, cp.newaxis] != 0) if GPU_AVAILABLE else \
                               np.divide(self.transition_matrix, row_sums[:, np.newaxis],
                                       where=row_sums[:, np.newaxis] != 0)
        
        # Calcula estatísticas para cada estado
        self.state_means = cp.zeros(self.n_states) if GPU_AVAILABLE else np.zeros(self.n_states)
        self.state_stds = cp.zeros(self.n_states) if GPU_AVAILABLE else np.zeros(self.n_states)
        for i in range(self.n_states):
            mask = states == i
            if cp.any(mask) if GPU_AVAILABLE else np.any(mask):
                self.state_means[i] = cp.mean(series[mask]) if GPU_AVAILABLE else np.mean(series[mask])
                self.state_stds[i] = cp.std(series[mask]) if GPU_AVAILABLE else np.std(series[mask])
        
        # Converte de volta para CPU se necessário
        if GPU_AVAILABLE:
            self.transition_matrix = cp.asnumpy(self.transition_matrix)
            self.state_means = cp.asnumpy(self.state_means)
            self.state_stds = cp.asnumpy(self.state_stds)
    
    def predict(self, n_steps: int, last_state: Optional[int] = None) -> np.ndarray:
        """Faz previsões usando a Cadeia de Markov."""
        if last_state is None:
            last_state = np.random.randint(0, self.n_states)
        
        predictions = []
        current_state = last_state
        
        for _ in range(n_steps):
            # Amostra o próximo estado baseado na matriz de transição
            next_state = np.random.choice(self.n_states, p=self.transition_matrix[current_state])
            
            # Gera previsão usando a distribuição do estado
            prediction = np.random.normal(self.state_means[next_state], self.state_stds[next_state])
            predictions.append(prediction)
            
            current_state = next_state
        
        return np.array(predictions)

class HiddenMarkovForecaster:
    def __init__(self, n_components: int = 3, n_iter: int = 100):
        self.n_components = n_components
        self.n_iter = n_iter
        self.model = hmm.GaussianHMM(n_components=n_components, n_iter=n_iter, random_state=42)
        self.scaler = None
        
    def fit(self, series: np.ndarray) -> None:
        """Treina o modelo HMM."""
        # Normaliza os dados usando GPU se disponível
        if GPU_AVAILABLE:
            series_gpu = cp.asarray(series)
            self.scaler = MinMaxScaler()
            normalized_series = cp.asnumpy(self.scaler.fit_transform(series_gpu.reshape(-1, 1)))
        else:
            self.scaler = MinMaxScaler()
            normalized_series = self.scaler.fit_transform(series.reshape(-1, 1))
        
        # Treina o modelo HMM
        self.model.fit(normalized_series)
    
    def predict(self, n_steps: int) -> np.ndarray:
        """Faz previsões usando o modelo HMM."""
        # Gera sequência de estados
        state_sequence = self.model.sample(n_steps)[0]
        
        # Desnormaliza as previsões usando GPU se disponível
        if GPU_AVAILABLE:
            state_sequence_gpu = cp.asarray(state_sequence)
            predictions = cp.asnumpy(self.scaler.inverse_transform(state_sequence_gpu))
        else:
            predictions = self.scaler.inverse_transform(state_sequence)
        
        return predictions.flatten()

def preparar_dados_forecast(caminho_arquivo: str) -> pd.DataFrame:
    """Prepara os dados para forecasting."""
    df = pd.read_csv(caminho_arquivo)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['Adj Close'])
    return df

def normalizar_serie(serie: np.ndarray) -> Tuple[np.ndarray, MinMaxScaler]:
    """Normaliza a série para o intervalo [0, 1] usando GPU se disponível."""
    scaler = MinMaxScaler()
    if GPU_AVAILABLE:
        serie_gpu = cp.asarray(serie)
        norm = cp.asnumpy(scaler.fit_transform(serie_gpu.reshape(-1, 1))).flatten()
    else:
        norm = scaler.fit_transform(serie.values.reshape(-1, 1)).flatten()
    return norm, scaler

def desnormalizar_serie(norm: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """Desfaz a normalização da série usando GPU se disponível."""
    if GPU_AVAILABLE:
        norm_gpu = cp.asarray(norm)
        return cp.asnumpy(scaler.inverse_transform(norm_gpu.reshape(-1, 1))).flatten()
    return scaler.inverse_transform(norm.reshape(-1, 1)).flatten()

def plotar_e_salvar(real: np.ndarray, previsto: np.ndarray, arquivo: str, ticker: str, modelo: str) -> None:
    """Plota e salva os resultados do forecasting."""
    plt.figure(figsize=(12, 6))
    plt.plot(real, label='Real', marker='o', alpha=0.7)
    plt.plot(previsto, label='Previsto', marker='x', alpha=0.7)
    plt.title(f'{arquivo} | {ticker} | {modelo}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    nome = os.path.join(PLOTS_DIR, f'plot_{arquivo.replace(",","_").replace(".csv","")}_{ticker}_{modelo}.png')
    plt.savefig(nome)
    plt.close()

def plotar_matriz_transicao(transition_matrix: np.ndarray, arquivo: str, ticker: str) -> None:
    """Plota e salva a matriz de transição."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(transition_matrix, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title(f'Matriz de Transição - {arquivo} | {ticker}')
    plt.tight_layout()
    nome = os.path.join(PLOTS_DIR, f'matriz_transicao_{arquivo.replace(",","_").replace(".csv","")}_{ticker}.png')
    plt.savefig(nome)
    plt.close()

def salvar_residuos(real: np.ndarray, previsto: np.ndarray, arquivo: str, ticker: str, modelo: str) -> None:
    """Salva os resíduos do modelo."""
    residuos = np.array(real) - np.array(previsto)
    df_res = pd.DataFrame({
        'real': real,
        'previsto': previsto,
        'residuo': residuos,
        'residuo_abs': np.abs(residuos)
    })
    nome = os.path.join(RESIDUOS_DIR, f'residuos_{arquivo.replace(",","_").replace(".csv","")}_{ticker}_{modelo}.csv')
    df_res.to_csv(nome, index=False)

def analisar_estados(modelo: HiddenMarkovForecaster, series: np.ndarray) -> Dict:
    """Analisa os estados do modelo HMM."""
    normalized_series = modelo.scaler.transform(series.reshape(-1, 1))
    states = modelo.model.predict(normalized_series)
    
    state_analysis = {}
    for i in range(modelo.n_components):
        mask = states == i
        if np.any(mask):
            state_analysis[f'Estado_{i}'] = {
                'media': np.mean(series[mask]),
                'std': np.std(series[mask]),
                'frequencia': np.mean(mask),
                'duracao_media': np.mean(np.diff(np.where(mask)[0]))
            }
    
    return state_analysis

# =====================
# EXECUCAO PRINCIPAL
# =====================

if __name__ == "__main__":
    logger.info("Iniciando pipeline de forecasting Markov...")
    if GPU_AVAILABLE:
        logger.info("Utilizando GPU para processamento (CuPy + TensorFlow)")
    else:
        logger.info("Utilizando CPU para processamento (NumPy)")

    arquivos = [f for f in os.listdir('dados_transformados') if f.endswith('.csv')]
    metricas = []
    previsoes = []
    analises_estados = []

    for arquivo in tqdm(arquivos, desc="Processando arquivos"):
        try:
            caminho = os.path.join('dados_transformados', arquivo)
            df = preparar_dados_forecast(caminho)

            if not all(col in df.columns for col in ['Date', 'Ticker', 'Adj Close']):
                logger.warning(f"{arquivo} não possui colunas obrigatórias. Pulando...")
                continue

            for ticker in tqdm(df['Ticker'].unique(), desc=f"{arquivo} - Tickers", leave=False):
                try:
                    logger.info(f"Iniciando {arquivo} | {ticker}")
                    df_ticker = df[df['Ticker'] == ticker].copy()
                    df_ticker.sort_values('Date', inplace=True)
                    series = df_ticker.set_index('Date')['Adj Close']

                    if len(series) < 100:
                        logger.warning(f"Ticker {ticker} com poucos dados. Ignorando...")
                        continue

                    # Normalização
                    norm_series, scaler = normalizar_serie(series)

                    # Markov Chain
                    try:
                        modelo_markov = MarkovChainForecaster(n_states=10)
                        modelo_markov.fit(norm_series)
                        pred_markov = modelo_markov.predict(n_steps=30)
                        pred_markov = desnormalizar_serie(pred_markov, scaler)
                        
                        # Plota matriz de transição
                        plotar_matriz_transicao(modelo_markov.transition_matrix, arquivo, ticker)
                    except Exception as e_markov:
                        logger.exception(f"Erro Markov Chain {arquivo} | {ticker}: {e_markov}")
                        modelo_markov = None
                        pred_markov = np.full(30, np.nan)

                    # Hidden Markov Model
                    try:
                        modelo_hmm = HiddenMarkovForecaster(n_components=3)
                        modelo_hmm.fit(norm_series)
                        pred_hmm = modelo_hmm.predict(n_steps=30)
                        pred_hmm = desnormalizar_serie(pred_hmm, scaler)
                        
                        # Análise dos estados
                        analise = analisar_estados(modelo_hmm, norm_series)
                        analises_estados.append({
                            'arquivo': arquivo,
                            'ticker': ticker,
                            'analise_estados': analise
                        })
                    except Exception as e_hmm:
                        logger.exception(f"Erro HMM {arquivo} | {ticker}: {e_hmm}")
                        modelo_hmm = None
                        pred_hmm = np.full(30, np.nan)

                    real = series[-30:].values
                    if len(real) == 30:
                        # Markov Chain
                        if modelo_markov is not None:
                            mse_markov = mean_squared_error(real, pred_markov)
                            mae_markov = mean_absolute_error(real, pred_markov)
                            metricas.append({
                                'arquivo': arquivo,
                                'ticker': ticker,
                                'modelo': 'MarkovChain',
                                'mse': mse_markov,
                                'mae': mae_markov
                            })
                            logger.info(f"{arquivo} | {ticker} | MarkovChain -> MSE: {mse_markov:.6f}, MAE: {mae_markov:.6f}")
                            previsoes.append({
                                'arquivo': arquivo,
                                'ticker': ticker,
                                'modelo': 'MarkovChain',
                                'real': real.tolist(),
                                'previsto': pred_markov.tolist()
                            })
                            plotar_e_salvar(real, pred_markov, arquivo, ticker, 'MarkovChain')
                            salvar_residuos(real, pred_markov, arquivo, ticker, 'MarkovChain')

                        # Hidden Markov Model
                        if modelo_hmm is not None:
                            mse_hmm = mean_squared_error(real, pred_hmm)
                            mae_hmm = mean_absolute_error(real, pred_hmm)
                            metricas.append({
                                'arquivo': arquivo,
                                'ticker': ticker,
                                'modelo': 'HMM',
                                'mse': mse_hmm,
                                'mae': mae_hmm
                            })
                            logger.info(f"{arquivo} | {ticker} | HMM -> MSE: {mse_hmm:.6f}, MAE: {mae_hmm:.6f}")
                            previsoes.append({
                                'arquivo': arquivo,
                                'ticker': ticker,
                                'modelo': 'HMM',
                                'real': real.tolist(),
                                'previsto': pred_hmm.tolist()
                            })
                            plotar_e_salvar(real, pred_hmm, arquivo, ticker, 'HMM')
                            salvar_residuos(real, pred_hmm, arquivo, ticker, 'HMM')

                    # Salvar modelos
                    if modelo_markov is not None:
                        pickle.dump({
                            'modelo': modelo_markov,
                            'scaler': scaler
                        }, open(os.path.join(MODELOS_DIR, f'{ticker}_markov.pkl'), 'wb'))
                    
                    if modelo_hmm is not None:
                        pickle.dump({
                            'modelo': modelo_hmm,
                            'scaler': scaler
                        }, open(os.path.join(MODELOS_DIR, f'{ticker}_hmm.pkl'), 'wb'))

                except Exception as e_ticker:
                    logger.exception(f"Erro ao processar ticker {ticker} em {arquivo}: {e_ticker}")

        except Exception as e:
            logger.exception(f"Erro ao processar {arquivo}: {e}")

    # Salvar métricas, previsões e análises
    pd.DataFrame(metricas).to_csv(os.path.join(METRICAS_DIR, 'metricas_forecasting_markov.csv'), index=False)
    pd.DataFrame(previsoes).to_csv(os.path.join(PREVISOES_DIR, 'previsoes_forecasting_markov.csv'), index=False)
    pd.DataFrame(analises_estados).to_csv(os.path.join(ANALISES_DIR, 'analises_estados_markov.csv'), index=False)
    logger.info("Pipeline de forecasting Markov finalizado!") 