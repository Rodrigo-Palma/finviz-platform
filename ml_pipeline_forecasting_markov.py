import matplotlib
matplotlib.use('Agg')  # Define o backend para evitar problemas com threads
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from loguru import logger
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from hmmlearn.hmm import GaussianHMM
from typing import Tuple

# =====================
# CONFIGURAÇÃO DE DIRETÓRIOS
# =====================
RESULTS_DIR = os.path.join('resultados', 'forecasting_markov')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
LOGS_DIR = 'logs'

for d in [RESULTS_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

logger.add(os.path.join(LOGS_DIR, 'forecast_pipeline_markov.log'), level='INFO', rotation='10 MB', encoding='utf-8')

# =====================
# FUNÇÕES AUXILIARES
# =====================

def mapear_volatilidade(volatilidade: float) -> str:
    if volatilidade < 0.02:
        return "BAIXA"
    elif 0.02 <= volatilidade < 0.05:
        return "MÉDIA"
    else:
        return "ALTA"

def identificar_regime(df_ticker: pd.DataFrame) -> Tuple[str, str]:
    try:
        if 'Volatility_5d' not in df_ticker.columns or 'Volatility_20d' not in df_ticker.columns:
            logger.warning("Colunas de volatilidade não encontradas. Usando apenas preços ajustados.")
            return "ERRO", "DESCONHECIDO"

        df_ticker = df_ticker.dropna(subset=['Adj Close', 'Volatility_5d', 'Volatility_20d'])

        serie = df_ticker['Adj Close'].values
        serie = serie[serie > 0]
        retornos = np.diff(np.log(serie)).reshape(-1, 1)
        volatilidade_5d = df_ticker['Volatility_5d'].iloc[1:].values.reshape(-1, 1)
        volatilidade_20d = df_ticker['Volatility_20d'].iloc[1:].values.reshape(-1, 1)

        observacoes = np.hstack([retornos, volatilidade_5d, volatilidade_20d])
        observacoes = observacoes[~np.isnan(observacoes).any(axis=1)]

        hmm = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=42)
        hmm.fit(observacoes)

        estados = hmm.predict(observacoes)
        regime = np.bincount(estados).argmax()

        if regime == 0:
            regime_desc = "TENDENCIA_BAIXA"
        elif regime == 1:
            regime_desc = "LATERAL"
        elif regime == 2:
            regime_desc = "TENDENCIA_ALTA"
        else:
            regime_desc = "DESCONHECIDO"

        media_volatilidade = np.mean(volatilidade_20d)
        categoria_volatilidade = mapear_volatilidade(media_volatilidade)

        return regime_desc, categoria_volatilidade
    except Exception as e:
        logger.error(f"Erro ao identificar regime de mercado: {e}")
        return "ERRO", "DESCONHECIDO"

def sanitize_filename(filename):
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def calcular_metricas(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}

def plotar_e_salvar(real: np.ndarray, previsto: np.ndarray, arquivo: str, ticker: str, modelo: str, metricas_hmm: dict) -> None:
    try:
        metricas_prev = calcular_metricas(real, previsto)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        ax1.plot(real, label='Valores Reais', color='blue')
        ax1.plot(previsto, label='Previsões', color='orange', linestyle='--')
        ax1.set_title(f'Previsão para {ticker} usando {modelo}')
        ax1.set_xlabel('Tempo')
        ax1.set_ylabel('Valor')
        ax1.legend()

        erro = real - previsto
        ax2.plot(erro, color='red', label='Erro de Previsão')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_title('Erro de Previsão ao Longo do Tempo')
        ax2.set_xlabel('Tempo')
        ax2.set_ylabel('Erro')
        ax2.legend()

        # Textos defensivos
        def format_metric(val):
            return f"{val:.2f}" if isinstance(val, (float, int)) else str(val)

        metricas_texto = [
            "Métricas de Previsão:",
            f"RMSE: {metricas_prev['RMSE']:.4f}",
            f"MAE: {metricas_prev['MAE']:.4f}",
            f"MAPE: {metricas_prev['MAPE']:.2f}%",
            "\nMétricas do Modelo HMM:",
            f"AIC: {format_metric(metricas_hmm.get('aic', 'N/A'))}",
            f"BIC: {format_metric(metricas_hmm.get('bic', 'N/A'))}",
            f"Log Likelihood: {format_metric(metricas_hmm.get('score', 'N/A'))}",
            f"Número de Estados: {metricas_hmm.get('n_components', 'N/A')}"
        ]

        plt.figtext(1.02, 0.5, '\n'.join(metricas_texto), fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

        nome_arquivo = sanitize_filename(f"plot_moedas_{ticker}_{modelo}.png")
        caminho_completo = os.path.join(PLOTS_DIR, nome_arquivo)
        plt.savefig(caminho_completo, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Gráfico salvo: {caminho_completo}")

    except Exception as e:
        logger.error(f"Erro ao salvar gráfico para {ticker}: {e}")

def preprocess_data(df_ticker: pd.DataFrame) -> Tuple[np.ndarray, bool]:
    try:
        df_clean = df_ticker.copy()
        df_clean = df_clean[df_clean['Volume'] > 0]
        df_clean = df_clean[df_clean['Adj Close'] > 0]

        log_adj_close = np.log(df_clean['Adj Close'].values)
        log_volume = np.log(df_clean['Volume'].values + 1)

        log_returns = np.diff(log_adj_close)
        log_volume_diff = np.diff(log_volume)

        log_returns_lag = log_returns[:-1]
        log_returns_main = log_returns[1:]
        log_volume_diff = log_volume_diff[1:]

        if len(log_returns) > 20:
            rolling_vol = pd.Series(log_returns).rolling(window=20).std().values[19:]
            min_len = min(len(log_returns_main), len(log_returns_lag), len(log_volume_diff), len(rolling_vol))
            features = np.column_stack([
                log_returns_main[-min_len:],
                log_returns_lag[-min_len:],
                log_volume_diff[-min_len:],
                rolling_vol[-min_len:]
            ])
        else:
            min_len = min(len(log_returns_main), len(log_returns_lag), len(log_volume_diff))
            features = np.column_stack([
                log_returns_main[-min_len:],
                log_returns_lag[-min_len:],
                log_volume_diff[-min_len:]
            ])

        features = features[~np.isnan(features).any(axis=1)]
        features = features[~np.isinf(features).any(axis=1)]

        return features, len(features) > 0
    except Exception as e:
        logger.error(f"Erro no pré-processamento dos dados: {e}")
        return np.array([]), False

def treinar_modelo_hmm(df_ticker: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, dict]:
    try:
        features, is_valid = preprocess_data(df_ticker)
        if not is_valid:
            raise ValueError("Dados insuficientes após o pré-processamento")

        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)

        train_size = int(len(features_scaled) * 0.8)
        train_data = features_scaled[:train_size]
        test_data = features_scaled[train_size:]

        best_model = None
        best_score = float('-inf')
        best_n_components = 2

        for n_components in range(2, 5):
            try:
                modelo = GaussianHMM(
                    n_components=n_components,
                    covariance_type="diag",
                    n_iter=1000,
                    random_state=42,
                    tol=1e-4,  # um pouco mais tolerante
                    verbose=False  # você pode colocar True para depurar
                )
                modelo.fit(train_data)

                # opcional: suavização da transmat_ para evitar warning
                modelo.transmat_ += 1e-6
                modelo.transmat_ /= modelo.transmat_.sum(axis=1, keepdims=True)

                score = modelo.score(test_data)

                if score > best_score:
                    best_score = score
                    best_model = modelo
                    best_n_components = n_components
            except Exception as e:
                logger.warning(f"Falha ao treinar HMM com {n_components} estados: {e}")
                continue

        if best_model is None:
            raise ValueError("Não foi possível treinar nenhum modelo HMM com sucesso")

        estados = best_model.predict(features_scaled)
        means = best_model.means_
        state_sequence = np.argmax(best_model.predict_proba(features_scaled), axis=1)

        previsoes = np.zeros(len(state_sequence))
        for i, estado in enumerate(state_sequence):
            previsoes[i] = means[estado, 0]

        adj_close_values = df_ticker['Adj Close'].values
        adj_close_values = adj_close_values[-len(previsoes):]

        precos_previstos = adj_close_values * np.exp(previsoes)

        # Cálculo manual do número de parâmetros
        N = best_model.n_components
        D = features_scaled.shape[1]
        n_params = (N - 1) + N * (N - 1) + N * D + N * D

        log_likelihood = best_model.score(features_scaled)
        n_samples = features_scaled.shape[0]

        aic = 2 * n_params - 2 * log_likelihood
        bic = np.log(n_samples) * n_params - 2 * log_likelihood

        metricas = {
            'aic': aic,
            'bic': bic,
            'score': best_score,
            'n_components': best_n_components,
            'rmse': np.sqrt(mean_squared_error(adj_close_values, precos_previstos)),
            'mae': mean_absolute_error(adj_close_values, precos_previstos)
        }

        return adj_close_values, precos_previstos, metricas

    except Exception as e:
        logger.error(f"Erro no treinamento do HMM: {e}")
        return np.array([]), np.array([]), {}

def processar_ticker(ticker, df_ticker, arquivo):
    try:
        logger.info(f"Iniciando {arquivo} | {ticker}")
        df_ticker.sort_values('Date', inplace=True)

        if len(df_ticker) < 100:
            logger.warning(f"Ticker {ticker} com poucos dados. Ignorando...")
            return None

        regime, volatilidade = identificar_regime(df_ticker)
        logger.info(f"Regime de mercado identificado: {regime}, Volatilidade: {volatilidade}")

        real, previsto, metricas_hmm = treinar_modelo_hmm(df_ticker)

        plotar_e_salvar(real, previsto, arquivo, ticker, f'HMM | {volatilidade}', metricas_hmm)

        return {
            'arquivo': arquivo,
            'ticker': ticker,
            'regime': regime,
            'volatilidade': volatilidade,
            'metricas_modelo': metricas_hmm,
            'real': real.tolist(),
            'previsto': previsto.tolist()
        }
    except Exception as e:
        logger.exception(f"Erro ao processar ticker {ticker} em {arquivo}: {e}")
        return None

if __name__ == "__main__":
    logger.info("Iniciando pipeline de forecasting Markov...")

    arquivos = [f for f in os.listdir('dados_transformados') if f.endswith('.csv')]
    resultados = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        for arquivo in tqdm(arquivos, desc="Processando arquivos"):
            try:
                caminho = os.path.join('dados_transformados', arquivo)
                df = pd.read_csv(caminho)
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df.dropna(subset=['Date', 'Adj Close'], inplace=True)

                futures = []
                for ticker in df['Ticker'].unique():
                    df_ticker = df[df['Ticker'] == ticker].copy()
                    futures.append(executor.submit(processar_ticker, ticker, df_ticker, arquivo))

                for future in futures:
                    result = future.result()
                    if result:
                        resultados.append(result)

            except Exception as e:
                logger.exception(f"Erro ao processar {arquivo}: {e}")

    resultados_df = pd.DataFrame(resultados)
    resultados_df.to_csv(os.path.join(RESULTS_DIR, 'resultados_forecasting.csv'), index=False)
    logger.info("Pipeline finalizado com sucesso!")
