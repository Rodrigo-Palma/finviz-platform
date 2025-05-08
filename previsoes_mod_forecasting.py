import os
import pickle
import pandas as pd
from statsmodels.tsa.arima.model import ARIMAResults
from prophet import Prophet

# ===============================
# FUNÇÕES PARA FAZER PREVISÃO
# ===============================

def carregar_modelo(caminho_modelo):
    modelo = pickle.load(open(caminho_modelo, 'rb'))
    return modelo

def prever_com_arima(modelo_arima, passos=5):
    previsao = modelo_arima.forecast(steps=passos)
    return previsao

def prever_com_prophet(modelo_prophet, passos=5):
    futuro = modelo_prophet.make_future_dataframe(periods=passos)
    previsao = modelo_prophet.predict(futuro)
    return previsao[['ds', 'yhat']].tail(passos)

# ===============================
# EXECUÇÃO PRINCIPAL
# ===============================

if __name__ == "__main__":
    pasta_modelos = 'modelos_forecasting'
    passos = 5

    previsoes_arima = []
    previsoes_prophet = []

    for arquivo in os.listdir(pasta_modelos):
        caminho = os.path.join(pasta_modelos, arquivo)

        if arquivo.endswith('_arima.pkl'):
            ticker = arquivo.replace('_arima.pkl', '')
            try:
                modelo_arima = carregar_modelo(caminho)
                previsao = prever_com_arima(modelo_arima, passos)

                # Gerar datas sequenciais (corrigir falta de index datetime)
                datas = pd.date_range(start=pd.Timestamp.today().normalize() + pd.Timedelta(days=1), periods=passos, freq='D')

                for data, valor in zip(datas, previsao):
                    previsoes_arima.append({'Ticker': ticker, 'Data': data.date(), 'Preco_Previsto_ARIMA': valor})

            except Exception as e:
                print(f"Erro ao prever ARIMA para {ticker}: {e}")

        elif arquivo.endswith('_prophet.pkl'):
            ticker = arquivo.replace('_prophet.pkl', '')
            try:
                modelo_prophet = carregar_modelo(caminho)
                previsao = prever_com_prophet(modelo_prophet, passos)

                for _, row in previsao.iterrows():
                    previsoes_prophet.append({'Ticker': ticker, 'Data': row['ds'].date(), 'Preco_Previsto_Prophet': row['yhat']})

            except Exception as e:
                print(f"Erro ao prever Prophet para {ticker}: {e}")

    # Salvar os resultados
    df_arima = pd.DataFrame(previsoes_arima)
    df_prophet = pd.DataFrame(previsoes_prophet)

    os.makedirs('logs/previsoes', exist_ok=True)
    df_arima.to_csv('logs/previsoes/previsoes_arima.csv', index=False)
    df_prophet.to_csv('logs/previsoes/previsoes_prophet.csv', index=False)

    print("\nPrevisões salvas em:")
    print("- logs/previsoes/previsoes_arima.csv")
    print("- logs/previsoes/previsoes_prophet.csv")
