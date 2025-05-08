import os
import pickle
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMAResults
from prophet import Prophet

# ===============================
# FUNCOES DE DESNORMALIZACAO
# ===============================

def desnormalizar(predicoes, minimo, maximo):
    return predicoes * (maximo - minimo) + minimo

# ===============================
# FUNCOES PARA FAZER PREVISAO
# ===============================

def carregar_modelo_e_normalizacao(ticker, modelo_tipo):
    caminho = f'modelos_forecasting/{ticker}_{modelo_tipo}.pkl'
    with open(caminho, 'rb') as f:
        obj = pickle.load(f)

    # Se salvamos o modelo como dict {'modelo': ..., 'min': ..., 'max': ...}
    if isinstance(obj, dict):
        modelo = obj['modelo']
        minimo = obj['min']
        maximo = obj['max']
    else:
        modelo = obj
        minimo = None
        maximo = None

    return modelo, minimo, maximo

def prever_com_arima(modelo_arima, passos=30):
    previsao = modelo_arima.forecast(steps=passos)
    return previsao

def prever_com_prophet(modelo_prophet, passos=30):
    futuro = modelo_prophet.make_future_dataframe(periods=passos)
    previsao = modelo_prophet.predict(futuro)
    return previsao[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(passos)

# ===============================
# PLOT COM PLOTLY
# ===============================

def plotar_previsoes(ticker, previsao_arima, previsao_prophet):
    primeira_data = previsao_prophet['ds'].iloc[0]
    datas_arima = pd.date_range(start=primeira_data, periods=len(previsao_arima), freq='D')

    fig = go.Figure()

    # Linha ARIMA
    fig.add_trace(go.Scatter(
        x=datas_arima,
        y=previsao_arima.values,
        mode='lines+markers',
        name='ARIMA Forecast'
    ))

    # Linha Prophet
    fig.add_trace(go.Scatter(
        x=previsao_prophet['ds'],
        y=previsao_prophet['yhat'],
        mode='lines+markers',
        name='Prophet Forecast'
    ))

    # Intervalo de confiança Prophet
    fig.add_trace(go.Scatter(
        x=pd.concat([previsao_prophet['ds'], previsao_prophet['ds'][::-1]]),
        y=pd.concat([previsao_prophet['yhat_upper'], previsao_prophet['yhat_lower'][::-1]]),
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=True,
        name='Intervalo Prophet'
    ))

    fig.update_layout(
        title=f'Previsão para {ticker}',
        xaxis_title='Data',
        yaxis_title='Preço Previsto',
        template='plotly_white'
    )

    fig.show()

    output_path = f'logs/previsoes/{ticker}_forecast.html'
    os.makedirs('logs/previsoes', exist_ok=True)
    fig.write_html(output_path)
    print(f"Gráfico salvo em: {output_path}")

# ===============================
# USO DO SCRIPT
# ===============================

if __name__ == "__main__":
    ticker = input("Digite o ticker para previsão (ex: ABEV3.SA): ").strip().upper()
    passos = int(input("Quantos dias à frente você quer prever? (ex: 30): "))

    arima_path = f'modelos_forecasting/{ticker}_arima.pkl'
    prophet_path = f'modelos_forecasting/{ticker}_prophet.pkl'

    if not (os.path.exists(arima_path) and os.path.exists(prophet_path)):
        print(f"Modelos para o ticker {ticker} não encontrados!")
    else:
        modelo_arima, minimo_arima, maximo_arima = carregar_modelo_e_normalizacao(ticker, 'arima')
        modelo_prophet, minimo_prophet, maximo_prophet = carregar_modelo_e_normalizacao(ticker, 'prophet')

        previsao_arima = prever_com_arima(modelo_arima, passos)
        previsao_prophet = prever_com_prophet(modelo_prophet, passos)

        if minimo_arima is not None and maximo_arima is not None:
            previsao_arima = desnormalizar(previsao_arima, minimo_arima, maximo_arima)

        if minimo_prophet is not None and maximo_prophet is not None:
            previsao_prophet['yhat'] = desnormalizar(previsao_prophet['yhat'], minimo_prophet, maximo_prophet)
            previsao_prophet['yhat_lower'] = desnormalizar(previsao_prophet['yhat_lower'], minimo_prophet, maximo_prophet)
            previsao_prophet['yhat_upper'] = desnormalizar(previsao_prophet['yhat_upper'], minimo_prophet, maximo_prophet)

        print("\n[ARIMA] Previsão:")
        print(previsao_arima)

        print("\n[Prophet] Previsão:")
        print(previsao_prophet)

        plotar_previsoes(ticker, previsao_arima, previsao_prophet)