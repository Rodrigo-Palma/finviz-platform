import os
import pandas as pd
import numpy as np
import warnings
import pickle
from tqdm import tqdm
from loguru import logger
from hmmlearn import hmm
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import Tuple, List, Dict, Optional
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats

# =====================
# TENTATIVA DE CUPY
# =====================
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    logger.info('CuPy disponível. Backend GPU ativado para operações NumPy.')
except ImportError:
    cp = np
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
# FUNCOES DE PREPROCESSAMENTO
# =====================

def identificar_categoria(arquivo: str) -> str:
    """Identifica a categoria do ativo baseado no nome do arquivo."""
    if 'criptos' in arquivo.lower():
        return 'CRYPTO'
    elif 'acoes_ibov' in arquivo.lower():
        return 'STOCK'
    elif 'moedas' in arquivo.lower():
        return 'CURRENCY'
    elif 'commodities' in arquivo.lower():
        return 'COMMODITY'
    elif 'juros' in arquivo.lower():
        return 'RATE'
    else:
        return 'OTHER'

def normalizar_por_categoria(serie: np.ndarray, categoria: str) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Versão melhorada da normalização por categoria.
    """
    if categoria == "CRYPTO":
        # Usar normalização robusta para criptomoedas
        scaler = RobustScaler()
    elif categoria == "CURRENCY":
        # Usar normalização padrão para moedas
        scaler = StandardScaler()
    else:
        # Usar MinMaxScaler para outros ativos
        scaler = MinMaxScaler()
    
    return scaler.fit_transform(serie.reshape(-1, 1)), scaler

def filtrar_outliers(serie: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Remove outliers usando Z-score com tratamento de NaN e método IQR.
    
    Args:
        serie: Array numpy com os valores
        threshold: Limiar para Z-score (default: 3.0)
    
    Returns:
        Array numpy com outliers removidos
    """
    # Remove valores NaN antes de calcular z-scores
    mask = ~np.isnan(serie)
    serie_limpa = serie[mask]
    
    if len(serie_limpa) == 0:
        return serie
    
    # Método Z-score
    z_scores = np.abs(stats.zscore(serie_limpa))
    mask_outliers_z = z_scores > threshold
    
    # Método IQR
    Q1 = np.percentile(serie_limpa, 25)
    Q3 = np.percentile(serie_limpa, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    mask_outliers_iqr = (serie_limpa < lower_bound) | (serie_limpa > upper_bound)
    
    # Combina os dois métodos
    mask_outliers = mask_outliers_z | mask_outliers_iqr
    
    # Cria array de saída com mesma forma do input
    resultado = serie.copy()
    resultado[mask] = np.where(mask_outliers, np.nan, serie_limpa)
    
    return resultado

def ajustar_pesos_por_regime(regime: str) -> Dict[str, float]:
    """Ajusta os pesos do ensemble baseado no regime de mercado."""
    if regime == 'ALTA_VOLATILIDADE':
        return {'markov': 0.2, 'hmm': 0.3, 'rf': 0.5}
    elif regime == 'TENDENCIA_ALTA':
        return {'markov': 0.3, 'hmm': 0.4, 'rf': 0.3}
    elif regime == 'TENDENCIA_BAIXA':
        return {'markov': 0.4, 'hmm': 0.3, 'rf': 0.3}
    elif regime == 'SOBREVENDIDO':
        return {'markov': 0.3, 'hmm': 0.2, 'rf': 0.5}
    elif regime == 'SOBRECOMPRADO':
        return {'markov': 0.3, 'hmm': 0.2, 'rf': 0.5}
    else:  # LATERAL
        return {'markov': 0.4, 'hmm': 0.3, 'rf': 0.3}

def validacao_cruzada_temporal(serie: np.ndarray, n_splits: int = 5) -> Tuple[float, float, float]:
    """
    Realiza validação cruzada temporal com tratamento de NaN e métricas adicionais.
    
    Args:
        serie: Array numpy com os valores
        n_splits: Número de splits para validação cruzada
    
    Returns:
        Tuple com MSE médio, desvio padrão e R² médio
    """
    # Remove valores NaN
    mask = ~np.isnan(serie)
    serie_limpa = serie[mask]
    
    if len(serie_limpa) < n_splits * 2:
        logger.warning(f"Dados insuficientes para validação cruzada. Retornando valores padrão.")
        return 1.0, 0.0, 0.0
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mse_scores = []
    r2_scores = []
    
    for train_idx, test_idx in tscv.split(serie_limpa):
        train_data = serie_limpa[train_idx]
        test_data = serie_limpa[test_idx]
        
        if len(train_data) < 10 or len(test_data) < 5:
            logger.warning(f"Fold com dados insuficientes. Pulando...")
            continue
        
        try:
            # Treina modelo
            modelo = MarkovChainForecaster()
            modelo.fit(train_data)
            
            # Faz previsões
            pred = modelo.predict(len(test_data))
            
            # Verifica se há NaN nas previsões
            if np.any(np.isnan(pred)):
                logger.warning(f"Previsões contêm NaN. Pulando fold...")
                continue
            
            # Calcula métricas
            mse = mean_squared_error(test_data, pred)
            r2 = 1 - (mse / np.var(test_data))
            
            mse_scores.append(mse)
            r2_scores.append(r2)
            
        except Exception as e:
            logger.warning(f"Erro no fold: {str(e)}. Pulando...")
            continue
    
    if not mse_scores:
        logger.warning("Nenhum fold válido para validação cruzada. Retornando valores padrão.")
        return 1.0, 0.0, 0.0
    
    return np.mean(mse_scores), np.std(mse_scores), np.mean(r2_scores)

def adicionar_features_especificas(df: pd.DataFrame, categoria: str) -> pd.DataFrame:
    """
    Adiciona features específicas por categoria com features técnicas avançadas.
    
    Args:
        df: DataFrame com os dados
        categoria: Categoria do ativo
    
    Returns:
        DataFrame com features adicionadas
    """
    # Garante que a coluna Date é datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    
    # Features comuns para todas as categorias
    df['Return_1d'] = df['Adj Close'].pct_change()
    df['Return_5d'] = df['Adj Close'].pct_change(5)
    df['Return_20d'] = df['Adj Close'].pct_change(20)
    df['Volatility_5d'] = df['Return_1d'].rolling(window=5).std()
    df['Volatility_20d'] = df['Return_1d'].rolling(window=20).std()
    
    # Médias móveis
    for window in [5, 20, 50, 100, 200]:
        df[f'SMA_{window}'] = df['Adj Close'].rolling(window=window).mean()
        df[f'EMA_{window}'] = df['Adj Close'].ewm(span=window, adjust=False).mean()
    
    # RSI
    delta = df['Adj Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Adj Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Adj Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['Bollinger_Upper'] = df['SMA_20'] + (df['Adj Close'].rolling(window=20).std() * 2)
    df['Bollinger_Lower'] = df['SMA_20'] - (df['Adj Close'].rolling(window=20).std() * 2)
    df['Bollinger_Width'] = (df['Bollinger_Upper'] - df['Bollinger_Lower']) / df['SMA_20']
    
    # ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Adj Close'].shift())
    low_close = np.abs(df['Low'] - df['Adj Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # OBV (On Balance Volume)
    df['OBV'] = (np.sign(df['Adj Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # Sharpe Ratio
    df['Sharpe_20d'] = (df['Return_20d'] / df['Volatility_20d']).fillna(0)
    
    if categoria == 'CRYPTO':
        # Features específicas para criptomoedas
        df['Volatilidade_1h'] = df['Return_1d'].rolling(window=24).std()
        df['Volume_MA'] = df['Volume'].rolling(window=24).mean()
        df['Volatilidade_Preco'] = df['Adj Close'].rolling(window=24).std()
        df['Volume_Price_Ratio'] = df['Volume'] / df['Adj Close']
        df['Market_Cap'] = df['Adj Close'] * df['Volume']
    
    elif categoria == 'STOCK':
        # Features específicas para ações
        df['P/L_Ratio'] = df['Adj Close'] / df['Volume']
        df['Razao_Volume_Preco'] = df['Volume'] / df['Adj Close']
        df['Momentum_Preco'] = df['Adj Close'].pct_change(5)
        df['Volume_MA_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        df['Price_MA_Ratio'] = df['Adj Close'] / df['SMA_20']
    
    elif categoria == 'CURRENCY':
        # Features específicas para moedas
        df['Diferencial_Taxa'] = df['Return_1d'].rolling(window=5).mean()
        df['Forca_Moeda'] = df['Adj Close'].rolling(window=20).mean()
        df['Razao_Volatilidade'] = df['Volatility_5d'] / df['Volatility_20d']
        df['Currency_Strength'] = df['Adj Close'] / df['SMA_50']
        df['Volatility_Ratio'] = df['Volatility_5d'] / df['Volatility_20d']
    
    elif categoria == 'COMMODITY':
        # Features específicas para commodities
        if isinstance(df.index, pd.DatetimeIndex):
            df['Fator_Sazonal'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
            df['Fator_Sazonal_Trimestral'] = np.sin(2 * np.pi * df.index.quarter / 4)
        else:
            df['Fator_Sazonal'] = 0
            df['Fator_Sazonal_Trimestral'] = 0
        df['Tendencia_Preco'] = df['Adj Close'].rolling(window=20).mean()
        df['Indice_Volatilidade'] = df['ATR'] / df['Adj Close']
        df['Price_Momentum'] = df['Adj Close'].pct_change(20)
        df['Volume_Trend'] = df['Volume'].pct_change(20)
    
    # Reseta o índice para manter Date como coluna
    df.reset_index(inplace=True)
    return df

# =====================
# FUNCOES DE FORECAST
# =====================

class EnsembleMarkovForecaster:
    def __init__(self, n_states: int = 3):
        self.n_states = n_states
        self.markov_chain = MarkovChainForecaster(n_states)
        self.hmm = HiddenMarkovForecaster(n_components=n_states)
        self.rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.weights = None
        self.scaler = None
        self.categoria = None
        self.feature_columns = [
            'SMA_5', 'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200',
            'EMA_5', 'EMA_20', 'EMA_50', 'EMA_100', 'EMA_200',
            'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'Bollinger_Upper', 'Bollinger_Lower', 'Bollinger_Width',
            'ATR', 'OBV', 'Return_1d', 'Return_5d', 'Return_20d',
            'Volatility_5d', 'Volatility_20d', 'Sharpe_20d'
        ]
    
    def fit(self, df: pd.DataFrame, categoria: str) -> None:
        """
        Treina o ensemble de modelos com otimização de pesos.
        
        Args:
            df: DataFrame com os dados
            categoria: Categoria do ativo
        """
        self.categoria = categoria
        
        # Adiciona features específicas
        df = adicionar_features_especificas(df, categoria)
        
        # Filtra outliers
        serie_limpa = filtrar_outliers(df['Adj Close'].values)
        df['Adj Close'] = serie_limpa
        
        # Remove linhas com NaN em colunas críticas
        colunas_criticas = ['Adj Close', 'Volume', 'Return_1d']
        df = df.dropna(subset=colunas_criticas)
        
        if len(df) < 100:
            raise ValueError("Dados insuficientes após limpeza")
        
        # Normaliza a série principal
        norm_series, self.scaler = normalizar_por_categoria(df['Adj Close'].values, categoria)
        
        # Verifica se há dados válidos após normalização
        if np.all(np.isnan(norm_series)):
            raise ValueError("Não há dados válidos após normalização")
        
        # Remove valores NaN da série normalizada
        mask = ~np.isnan(norm_series)
        norm_series_clean = norm_series[mask]
        
        # Treina modelos base
        self.markov_chain.fit(norm_series_clean)
        self.hmm.fit(norm_series_clean)
        
        # Prepara features para Random Forest
        X = df[self.feature_columns].fillna(0).values
        y = df['Adj Close'].values
        
        # Verifica se há dados suficientes para treinar o RF
        if len(X) < 10:
            raise ValueError("Dados insuficientes para treinar Random Forest")
        
        # Verifica se há NaN nas features
        if np.any(np.isnan(X)):
            logger.warning("Features contêm NaN. Preenchendo com 0...")
            X = np.nan_to_num(X, nan=0.0)
        
        # Verifica se há NaN no target
        if np.any(np.isnan(y)):
            logger.warning("Target contém NaN. Removendo linhas...")
            mask = ~np.isnan(y)
            X = X[mask]
            y = y[mask]
        
        # Treina Random Forest
        self.rf.fit(X, y)
        
        # Calcula pesos baseado no erro recente e importância das features
        markov_error = self._calcular_erro_recente(self.markov_chain, norm_series_clean)
        hmm_error = self._calcular_erro_recente(self.hmm, norm_series_clean)
        rf_error = self._calcular_erro_rf(self.rf, X, y)
        
        # Ajusta pesos considerando a importância das features
        rf_importance = np.mean(self.rf.feature_importances_)
        total_error = markov_error + hmm_error + rf_error
        
        self.weights = {
            'markov': (hmm_error / total_error) * (1 - rf_importance),
            'hmm': (markov_error / total_error) * (1 - rf_importance),
            'rf': rf_error / total_error
        }
        
        # Normaliza os pesos para somarem 1
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
    
    def _calcular_erro_recente(self, modelo, series: np.ndarray, window: int = 30) -> float:
        """
        Calcula o erro recente do modelo com janela móvel.
        
        Args:
            modelo: Modelo a ser avaliado
            series: Série temporal
            window: Tamanho da janela móvel
        
        Returns:
            Erro médio recente
        """
        if len(series) < window:
            return 1.0
        
        recent_series = series[-window:]
        pred = modelo.predict(window)
        
        # Calcula erro com diferentes métricas
        mse = mean_squared_error(recent_series, pred)
        mae = mean_absolute_error(recent_series, pred)
        
        # Combina as métricas
        return 0.7 * mse + 0.3 * mae
    
    def _calcular_erro_rf(self, modelo, X: np.ndarray, y: np.ndarray, window: int = 30) -> float:
        """
        Calcula o erro recente do Random Forest com validação cruzada.
        
        Args:
            modelo: Modelo Random Forest
            X: Features
            y: Target
            window: Tamanho da janela móvel
        
        Returns:
            Erro médio recente
        """
        if len(X) < window:
            return 1.0
        
        recent_X = X[-window:]
        recent_y = y[-window:]
        pred = modelo.predict(recent_X)
        
        # Calcula erro com diferentes métricas
        mse = mean_squared_error(recent_y, pred)
        mae = mean_absolute_error(recent_y, pred)
        
        # Combina as métricas
        return 0.7 * mse + 0.3 * mae
    
    def predict(self, df: pd.DataFrame, n_steps: int = 30) -> np.ndarray:
        """
        Faz previsões usando o ensemble com otimização de pesos.
        
        Args:
            df: DataFrame com os dados
            n_steps: Número de passos para prever
        
        Returns:
            Array numpy com as previsões
        """
        # Adiciona features específicas
        df = adicionar_features_especificas(df, self.categoria)
        
        # Previsões dos modelos base
        norm_series = self.scaler.transform(df['Adj Close'].values.reshape(-1, 1)).flatten()
        pred_markov = self.markov_chain.predict(n_steps)
        pred_hmm = self.hmm.predict(n_steps)
        
        # Previsão do Random Forest
        X = df[self.feature_columns].fillna(0).values
        pred_rf = self.rf.predict(X[-n_steps:])
        
        # Identifica regime e ajusta pesos
        regime = self.identificar_regime_mercado(df)
        pesos_ajustados = ajustar_pesos_por_regime(regime)
        
        # Combina previsões com pesos ajustados
        pred_ensemble = (
            pesos_ajustados['markov'] * pred_markov +
            pesos_ajustados['hmm'] * pred_hmm +
            pesos_ajustados['rf'] * pred_rf
        )
        
        # Aplica suavização exponencial
        alpha = 0.3  # Fator de suavização
        pred_suavizada = np.zeros_like(pred_ensemble)
        pred_suavizada[0] = pred_ensemble[0]
        for i in range(1, len(pred_ensemble)):
            pred_suavizada[i] = alpha * pred_ensemble[i] + (1 - alpha) * pred_suavizada[i-1]
        
        # Desnormaliza a previsão final
        return self.scaler.inverse_transform(pred_suavizada.reshape(-1, 1)).flatten()
    
    def identificar_regime_mercado(self, df: pd.DataFrame) -> str:
        """
        Identifica o regime de mercado atual com métricas avançadas.
        
        Args:
            df: DataFrame com os dados
        
        Returns:
            String indicando o regime de mercado
        """
        returns = df['Return_20d'].iloc[-1]
        vol = df['Volatility_20d'].iloc[-1]
        rsi = df['RSI_14'].iloc[-1]
        macd = df['MACD'].iloc[-1]
        bb_width = df['Bollinger_Width'].iloc[-1]
        
        # Identifica tendência
        sma_20 = df['SMA_20'].iloc[-1]
        sma_50 = df['SMA_50'].iloc[-1]
        sma_200 = df['SMA_200'].iloc[-1]
        
        # Identifica força do mercado
        adx = df['ATR'].iloc[-1] / df['Volatility_20d'].iloc[-1]
        
        if vol > 0.05 and adx > 1.5:
            return "ALTA_VOLATILIDADE"
        elif returns > 0.02 and sma_20 > sma_50 and sma_50 > sma_200:
            return "TENDENCIA_ALTA"
        elif returns < -0.02 and sma_20 < sma_50 and sma_50 < sma_200:
            return "TENDENCIA_BAIXA"
        elif rsi > 70 and macd < 0:
            return "SOBREVENDIDO"
        elif rsi < 30 and macd > 0:
            return "SOBRECOMPRADO"
        elif bb_width < 0.1:
            return "LATERAL_ESTRITO"
        else:
            return "LATERAL"

class MarkovChainForecaster:
    def __init__(self, n_states: int = 3, n_iter: int = 100):
        """
        Inicializa o modelo de Cadeia de Markov.
        
        Args:
            n_states (int): Número de estados no modelo (padrão: 3)
            n_iter (int): Número máximo de iterações para treinamento
        """
        self.n_states = n_states
        self.n_iter = n_iter
        self.transition_matrix = None
        self.state_means = None
        self.state_stds = None
        self.state_probs = None
        self.discretizer = KBinsDiscretizer(n_bins=n_states, encode='ordinal', strategy='quantile')
        
    def fit(self, series: np.ndarray) -> None:
        """
        Treina o modelo de Cadeia de Markov com otimização de parâmetros.
        
        Args:
            series: Array numpy com os valores
        """
        # Remove valores NaN antes da discretização
        mask = ~np.isnan(series)
        series_clean = series[mask]
        
        if len(series_clean) == 0:
            raise ValueError("Não há dados válidos para treinar o modelo")
        
        # Converte para GPU se disponível
        if GPU_AVAILABLE:
            series_clean = cp.asarray(series_clean)
        
        # Discretiza a série em estados
        states = self.discretizer.fit_transform(series_clean.reshape(-1, 1)).flatten()
        
        # Calcula a matriz de transição com suavização
        self.transition_matrix = cp.zeros((self.n_states, self.n_states)) if GPU_AVAILABLE else np.zeros((self.n_states, self.n_states))
        for i in range(len(states) - 1):
            self.transition_matrix[int(states[i]), int(states[i + 1])] += 1
        
        # Suavização de Laplace
        alpha = 0.1  # Parâmetro de suavização
        self.transition_matrix += alpha
        
        # Normaliza a matriz de transição
        row_sums = self.transition_matrix.sum(axis=1)
        self.transition_matrix = cp.divide(self.transition_matrix, row_sums[:, cp.newaxis],
                                         where=row_sums[:, cp.newaxis] != 0) if GPU_AVAILABLE else \
                               np.divide(self.transition_matrix, row_sums[:, np.newaxis],
                                       where=row_sums[:, np.newaxis] != 0)
        
        # Calcula estatísticas para cada estado
        self.state_means = cp.zeros(self.n_states) if GPU_AVAILABLE else np.zeros(self.n_states)
        self.state_stds = cp.zeros(self.n_states) if GPU_AVAILABLE else np.zeros(self.n_states)
        self.state_probs = cp.zeros(self.n_states) if GPU_AVAILABLE else np.zeros(self.n_states)
        
        for i in range(self.n_states):
            mask_state = states == i
            if cp.any(mask_state) if GPU_AVAILABLE else np.any(mask_state):
                self.state_means[i] = cp.mean(series_clean[mask_state]) if GPU_AVAILABLE else np.mean(series_clean[mask_state])
                self.state_stds[i] = cp.std(series_clean[mask_state]) if GPU_AVAILABLE else np.std(series_clean[mask_state])
                self.state_probs[i] = cp.mean(mask_state) if GPU_AVAILABLE else np.mean(mask_state)
        
        # Converte de volta para CPU se necessário
        if GPU_AVAILABLE:
            self.transition_matrix = cp.asnumpy(self.transition_matrix)
            self.state_means = cp.asnumpy(self.state_means)
            self.state_stds = cp.asnumpy(self.state_stds)
            self.state_probs = cp.asnumpy(self.state_probs)
    
    def predict(self, n_steps: int, last_state: Optional[int] = None) -> np.ndarray:
        """
        Faz previsões usando a Cadeia de Markov com suavização.
        
        Args:
            n_steps: Número de passos para prever
            last_state: Estado inicial (opcional)
        
        Returns:
            Array numpy com as previsões
        """
        if last_state is None:
            last_state = np.random.choice(self.n_states, p=self.state_probs)
        
        predictions = []
        current_state = last_state
        
        for _ in range(n_steps):
            # Amostra o próximo estado baseado na matriz de transição
            next_state = np.random.choice(self.n_states, p=self.transition_matrix[current_state])
            
            # Gera previsão usando a distribuição do estado com suavização
            prediction = np.random.normal(self.state_means[next_state], self.state_stds[next_state])
            
            # Suaviza a previsão usando a média dos estados vizinhos
            if next_state > 0 and next_state < self.n_states - 1:
                prediction = 0.7 * prediction + 0.15 * self.state_means[next_state - 1] + 0.15 * self.state_means[next_state + 1]
            
            predictions.append(prediction)
            current_state = next_state
        
        return np.array(predictions)

class HiddenMarkovForecaster:
    def __init__(self, n_components: int = 3, n_iter: int = 100):
        """
        Inicializa o modelo HMM (Hidden Markov Model).
        
        Args:
            n_components (int): Número de estados ocultos (padrão: 3)
            n_iter (int): Número máximo de iterações para treinamento
        """
        self.n_components = n_components
        self.n_iter = n_iter
        self.model = hmm.GaussianHMM(
            n_components=n_components,
            n_iter=n_iter,
            random_state=42,
            covariance_type="full",
            init_params="kmeans"
        )
        self.scaler = None
        self.state_means = None
        self.state_covars = None
        self.transmat = None
        
    def fit(self, series: np.ndarray) -> None:
        """
        Treina o modelo HMM com otimização de parâmetros.
        
        Args:
            series: Array numpy com os valores
        """
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
        
        # Armazena parâmetros do modelo
        self.state_means = self.model.means_
        self.state_covars = self.model.covars_
        self.transmat = self.model.transmat_
        
        # Aplica suavização na matriz de transição
        alpha = 0.1
        self.transmat = (1 - alpha) * self.transmat + alpha / self.n_components
    
    def predict(self, n_steps: int) -> np.ndarray:
        """
        Faz previsões usando o modelo HMM com suavização.
        
        Args:
            n_steps: Número de passos para prever
        
        Returns:
            Array numpy com as previsões
        """
        # Gera sequência de estados
        state_sequence = self.model.sample(n_steps)[0]
        
        # Aplica suavização nas previsões
        predictions = np.zeros(n_steps)
        for i in range(n_steps):
            # Converte o estado para inteiro
            state = int(state_sequence[i])
            
            # Usa média ponderada dos estados vizinhos
            if state > 0 and state < self.n_components - 1:
                predictions[i] = 0.7 * self.state_means[state][0] + \
                               0.15 * self.state_means[state - 1][0] + \
                               0.15 * self.state_means[state + 1][0]
            else:
                predictions[i] = self.state_means[state][0]
        
        # Desnormaliza as previsões usando GPU se disponível
        if GPU_AVAILABLE:
            predictions_gpu = cp.asarray(predictions)
            predictions = cp.asnumpy(self.scaler.inverse_transform(predictions_gpu.reshape(-1, 1)))
        else:
            predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1))
        
        return predictions.flatten()

def preparar_dados_forecast(caminho_arquivo: str) -> pd.DataFrame:
    """Prepara os dados para forecasting com tratamento robusto de NaN."""
    df = pd.read_csv(caminho_arquivo)
    
    # Converte coluna Date para datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
    
    # Remove linhas com NaN em colunas críticas
    colunas_criticas = ['Date', 'Ticker', 'Adj Close']
    df = df.dropna(subset=colunas_criticas)
    
    # Substitui inf/-inf por NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Preenche NaN em colunas numéricas com forward fill e backward fill
    colunas_numericas = df.select_dtypes(include=[np.number]).columns
    df[colunas_numericas] = df[colunas_numericas].fillna(method='ffill').fillna(method='bfill')
    
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
    """
    Analisa os estados do modelo HMM com tratamento de NaN.
    
    Args:
        modelo: Modelo HMM treinado
        series: Série temporal de preços
        
    Returns:
        Dict: Dicionário com análise de cada estado
    """
    # Remove valores NaN
    mask = ~np.isnan(series)
    series_clean = series[mask]
    
    if len(series_clean) == 0:
        logger.warning("Não há dados válidos para análise de estados")
        return {}
    
    try:
        normalized_series = modelo.scaler.transform(series_clean.reshape(-1, 1))
        states = modelo.model.predict(normalized_series)
        
        state_analysis = {}
        for i in range(modelo.n_components):
            mask = states == i
            if np.any(mask):
                state_analysis[f'Estado_{i}'] = {
                    'media': float(np.mean(series_clean[mask])),
                    'std': float(np.std(series_clean[mask])),
                    'frequencia': float(np.mean(mask)),
                    'duracao_media': float(np.mean(np.diff(np.where(mask)[0]))) if len(np.where(mask)[0]) > 1 else 0.0,
                    'interpretacao': interpretar_estado(float(np.mean(series_clean[mask])), float(np.std(series_clean[mask])))
                }
        
        return state_analysis
    except Exception as e:
        logger.warning(f"Erro na análise de estados: {str(e)}")
        return {}

def interpretar_estado(media: float, std: float) -> str:
    """
    Interpreta o estado baseado em suas estatísticas.
    
    Args:
        media: Média do estado
        std: Desvio padrão do estado
        
    Returns:
        str: Interpretação do estado
    """
    if std < 0.1:
        return "Estado de Baixa Volatilidade"
    elif std < 0.2:
        return "Estado de Volatilidade Média"
    else:
        return "Estado de Alta Volatilidade"

def avaliar_modelo(real: np.ndarray, previsto: np.ndarray) -> Dict[str, float]:
    """
    Avalia o modelo usando múltiplas métricas.
    
    Args:
        real: Array numpy com valores reais
        previsto: Array numpy com valores previstos
    
    Returns:
        Dicionário com métricas de avaliação
    """
    # Remove valores NaN
    mask = ~np.isnan(real) & ~np.isnan(previsto)
    real_clean = real[mask]
    previsto_clean = previsto[mask]
    
    if len(real_clean) == 0:
        return {
            'mse': 1.0,
            'rmse': 1.0,
            'mae': 1.0,
            'mape': 100.0,
            'r2': 0.0,
            'direction_accuracy': 0.0
        }
    
    # Métricas básicas
    mse = mean_squared_error(real_clean, previsto_clean)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(real_clean, previsto_clean)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((real_clean - previsto_clean) / real_clean)) * 100
    
    # R²
    r2 = 1 - (mse / np.var(real_clean))
    
    # Acurácia de direção
    real_direction = np.diff(real_clean)
    pred_direction = np.diff(previsto_clean)
    direction_accuracy = np.mean(np.sign(real_direction) == np.sign(pred_direction))
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'direction_accuracy': direction_accuracy
    }

def plotar_metricas(metricas: pd.DataFrame, arquivo: str) -> None:
    """
    Plota e salva as métricas de avaliação.
    
    Args:
        metricas: DataFrame com as métricas
        arquivo: Nome do arquivo para salvar o plot
    """
    plt.figure(figsize=(15, 10))
    
    # MSE por categoria
    plt.subplot(2, 2, 1)
    sns.boxplot(x='categoria', y='mse', data=metricas)
    plt.title('MSE por Categoria')
    plt.xticks(rotation=45)
    
    # R² por categoria
    plt.subplot(2, 2, 2)
    sns.boxplot(x='categoria', y='r2', data=metricas)
    plt.title('R² por Categoria')
    plt.xticks(rotation=45)
    
    # Acurácia de direção por categoria
    plt.subplot(2, 2, 3)
    sns.boxplot(x='categoria', y='direction_accuracy', data=metricas)
    plt.title('Acurácia de Direção por Categoria')
    plt.xticks(rotation=45)
    
    # MAPE por categoria
    plt.subplot(2, 2, 4)
    sns.boxplot(x='categoria', y='mape', data=metricas)
    plt.title('MAPE por Categoria')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'metricas_{arquivo}.png'))
    plt.close()

def plotar_residuos(real: np.ndarray, previsto: np.ndarray, arquivo: str, ticker: str, modelo: str) -> None:
    """
    Plota e salva análise de resíduos.
    
    Args:
        real: Array numpy com valores reais
        previsto: Array numpy com valores previstos
        arquivo: Nome do arquivo
        ticker: Ticker do ativo
        modelo: Nome do modelo
    """
    residuos = real - previsto
    
    plt.figure(figsize=(15, 10))
    
    # Resíduos vs Valores Previstos
    plt.subplot(2, 2, 1)
    plt.scatter(previsto, residuos, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Valores Previstos')
    plt.ylabel('Resíduos')
    plt.title('Resíduos vs Valores Previstos')
    
    # Histograma dos Resíduos
    plt.subplot(2, 2, 2)
    sns.histplot(residuos, kde=True)
    plt.xlabel('Resíduos')
    plt.ylabel('Frequência')
    plt.title('Distribuição dos Resíduos')
    
    # QQ-Plot
    plt.subplot(2, 2, 3)
    stats.probplot(residuos, dist="norm", plot=plt)
    plt.title('QQ-Plot dos Resíduos')
    
    # Resíduos vs Tempo
    plt.subplot(2, 2, 4)
    plt.plot(residuos, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Tempo')
    plt.ylabel('Resíduos')
    plt.title('Resíduos vs Tempo')
    
    plt.suptitle(f'Análise de Resíduos - {arquivo} | {ticker} | {modelo}')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'residuos_{arquivo}_{ticker}_{modelo}.png'))
    plt.close()

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
            categoria = identificar_categoria(arquivo)

            if not all(col in df.columns for col in ['Date', 'Ticker', 'Adj Close']):
                logger.warning(f"{arquivo} não possui colunas obrigatórias. Pulando...")
                continue

            for ticker in tqdm(df['Ticker'].unique(), desc=f"{arquivo} - Tickers", leave=False):
                try:
                    logger.info(f"Iniciando {arquivo} | {ticker}")
                    df_ticker = df[df['Ticker'] == ticker].copy()
                    df_ticker.sort_values('Date', inplace=True)

                    if len(df_ticker) < 100:
                        logger.warning(f"Ticker {ticker} com poucos dados. Ignorando...")
                        continue

                    # Modelo Ensemble
                    try:
                        modelo_ensemble = EnsembleMarkovForecaster()
                        modelo_ensemble.fit(df_ticker, categoria)
                        
                        # Validação cruzada temporal
                        cv_mean, cv_std, cv_r2 = validacao_cruzada_temporal(df_ticker['Adj Close'].values)
                        logger.info(f"Validação Cruzada - MSE médio: {cv_mean:.6f}, std: {cv_std:.6f}, R² médio: {cv_r2:.6f}")
                        
                        pred_ensemble = modelo_ensemble.predict(df_ticker)
                        
                        # Identifica regime de mercado
                        regime = modelo_ensemble.identificar_regime_mercado(df_ticker)
                        logger.info(f"Regime de mercado identificado: {regime}")
                        
                        # Análise dos estados
                        analise = analisar_estados(modelo_ensemble.hmm, df_ticker['Adj Close'].values)
                        analises_estados.append({
                            'arquivo': arquivo,
                            'ticker': ticker,
                            'categoria': categoria,
                            'regime': regime,
                            'cv_mean': cv_mean,
                            'cv_std': cv_std,
                            'cv_r2': cv_r2,
                            'analise_estados': analise
                        })
                    except Exception as e_ensemble:
                        logger.exception(f"Erro Ensemble {arquivo} | {ticker}: {e_ensemble}")
                        modelo_ensemble = None
                        pred_ensemble = np.full(30, np.nan)

                    real = df_ticker['Adj Close'].values[-30:]
                    if len(real) == 30 and not np.any(np.isnan(real)):
                        # Ensemble
                        if modelo_ensemble is not None and not np.any(np.isnan(pred_ensemble)):
                            # Avalia o modelo
                            metricas_modelo = avaliar_modelo(real, pred_ensemble)
                            
                            # Adiciona métricas ao DataFrame
                            metricas.append({
                                'arquivo': arquivo,
                                'ticker': ticker,
                                'categoria': categoria,
                                'modelo': 'Ensemble',
                                'regime': regime,
                                'cv_mean': cv_mean,
                                'cv_std': cv_std,
                                'cv_r2': cv_r2,
                                **metricas_modelo
                            })
                            
                            logger.info(f"{arquivo} | {ticker} | Ensemble -> MSE: {metricas_modelo['mse']:.6f}, "
                                      f"MAE: {metricas_modelo['mae']:.6f}, R²: {metricas_modelo['r2']:.6f}, "
                                      f"Direção: {metricas_modelo['direction_accuracy']:.2%}")
                            
                            previsoes.append({
                                'arquivo': arquivo,
                                'ticker': ticker,
                                'categoria': categoria,
                                'modelo': 'Ensemble',
                                'regime': regime,
                                'real': real.tolist(),
                                'previsto': pred_ensemble.tolist()
                            })
                            
                            # Plota resultados
                            plotar_e_salvar(real, pred_ensemble, arquivo, ticker, 'Ensemble')
                            plotar_residuos(real, pred_ensemble, arquivo, ticker, 'Ensemble')

                    # Salvar modelos
                    if modelo_ensemble is not None:
                        pickle.dump({
                            'modelo': modelo_ensemble,
                            'scaler': modelo_ensemble.scaler,
                            'categoria': categoria
                        }, open(os.path.join(MODELOS_DIR, f'{ticker}_ensemble.pkl'), 'wb'))

                except Exception as e_ticker:
                    logger.exception(f"Erro ao processar ticker {ticker} em {arquivo}: {e_ticker}")

        except Exception as e:
            logger.exception(f"Erro ao processar {arquivo}: {e}")

    # Salvar métricas, previsões e análises
    df_metricas = pd.DataFrame(metricas)
    df_metricas.to_csv(os.path.join(METRICAS_DIR, 'metricas_forecasting_markov.csv'), index=False)
    pd.DataFrame(previsoes).to_csv(os.path.join(PREVISOES_DIR, 'previsoes_forecasting_markov.csv'), index=False)
    pd.DataFrame(analises_estados).to_csv(os.path.join(ANALISES_DIR, 'analises_estados_markov.csv'), index=False)
    
    # Plota métricas gerais
    plotar_metricas(df_metricas, 'forecasting_markov')
    
    logger.info("Pipeline de forecasting Markov finalizado!") 