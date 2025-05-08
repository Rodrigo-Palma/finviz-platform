import os
import re
import pandas as pd

# ==============================
# SCRIPT PARA CALCULAR PROBABILIDADES
# ==============================

def carregar_metricas(log_path):
    """Extrai MSE e MAE de cada modelo do log."""
    metricas = []
    with open(log_path, 'r', encoding='utf-8') as f:
        linhas = f.readlines()

    for linha in linhas:
        match_arima = re.search(r'(.+?) \| (.+?) \| ARIMA -> MSE: ([0-9.]+), MAE: ([0-9.]+)', linha)
        match_prophet = re.search(r'(.+?) \| (.+?) \| Prophet -> MSE: ([0-9.]+), MAE: ([0-9.]+)', linha)

        if match_arima:
            arquivo, ticker, mse, mae = match_arima.groups()
            metricas.append({'Ticker': ticker, 'Modelo': 'ARIMA', 'MSE': float(mse), 'MAE': float(mae)})

        if match_prophet:
            arquivo, ticker, mse, mae = match_prophet.groups()
            metricas.append({'Ticker': ticker, 'Modelo': 'Prophet', 'MSE': float(mse), 'MAE': float(mae)})

    return pd.DataFrame(metricas)

def calcular_probabilidades(df_metricas):
    """Calcula probabilidade baseado no MSE."""
    df_metricas['Probabilidade'] = 1 / (1 + df_metricas['MSE'])
    return df_metricas

# ==============================
# EXECUCAO PRINCIPAL
# ==============================

if __name__ == "__main__":
    log_path = 'logs/forecast_pipeline.log'

    if not os.path.exists(log_path):
        print("Log forecast_pipeline.log nao encontrado!")
        exit()

    df_metricas = carregar_metricas(log_path)
    df_metricas = calcular_probabilidades(df_metricas)

    # Pivotar para juntar ARIMA e Prophet lado a lado
    df_pivot = df_metricas.pivot(index='Ticker', columns='Modelo', values='Probabilidade').reset_index()
    df_pivot.columns = ['Ticker', 'Probabilidade_ARIMA', 'Probabilidade_Prophet']

    # Melhor modelo baseado na maior probabilidade
    df_pivot['Melhor_Modelo'] = df_pivot[['Probabilidade_ARIMA', 'Probabilidade_Prophet']].idxmax(axis=1)
    df_pivot['Melhor_Modelo'] = df_pivot['Melhor_Modelo'].apply(lambda x: x.replace('Probabilidade_', ''))

    os.makedirs('logs/previsoes', exist_ok=True)
    df_pivot.to_csv('logs/previsoes/probs_modelos.csv', index=False)

    print("\nProbabilidades calculadas e salvas em logs/previsoes/probs_modelos.csv")
