import os
import requests
import pandas as pd
import re
import time
from urllib.parse import quote
from datetime import datetime, timedelta
from tqdm import tqdm
from loguru import logger

# ============================================
# CONFIGURAÇÃO DO LOGGER
# ============================================
os.makedirs('logs', exist_ok=True)
logger.add('logs/execucao.log', level='INFO', rotation='10 MB', encoding='utf-8', enqueue=True)
logger.add('logs/erros.log', level='ERROR', rotation='10 MB', encoding='utf-8', enqueue=True)

# ============================================
# GARANTE QUE AS PASTAS EXISTAM
# ============================================
def garantir_pastas():
    os.makedirs('dados', exist_ok=True)

# ============================================
# FUNÇÃO PARA COLETAR DADOS DO YAHOO
# ============================================
def obter_dados_historicos_yahoo(ticker: str, inicio: datetime, fim: datetime, intervalo: str = '1d', tentativas=5) -> pd.DataFrame:
    sessao = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7"
    }

    ticker_esc = quote(ticker, safe='')
    url_hist = f"https://finance.yahoo.com/quote/{ticker_esc}/history"
    sessao.get(url_hist, headers=headers)

    inicio_ts = int(inicio.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
    fim_ts = int(fim.replace(hour=23, minute=59, second=59).timestamp())

    url = (
        f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker_esc}"
        f"?period1={inicio_ts}&period2={fim_ts}&interval={intervalo}"
        f"&events=history&includeAdjustedClose=true"
    )

    espera = 2
    for tentativa in range(tentativas):
        resp = sessao.get(url, headers=headers)
        
        if resp.status_code == 200:
            data = resp.json()
            chart = data.get('chart', {})
            result = chart.get('result')
            if not result:
                err = chart.get('error')
                logger.error(f"Erro API para {ticker}: {err}")
                return None

            result = result[0]
            timestamps = result.get('timestamp', [])
            if not timestamps:
                logger.error(f"Sem timestamps para {ticker}")
                return None

            indicators = result.get('indicators', {}).get('quote', [{}])[0]
            opens = indicators.get('open', [])
            highs = indicators.get('high', [])
            lows = indicators.get('low', [])
            closes = indicators.get('close', [])
            volumes = indicators.get('volume', [])

            adjclose_arr = result.get('indicators', {}).get('adjclose', [{}])[0].get('adjclose', closes)

            dates = [datetime.fromtimestamp(ts).replace(hour=0, minute=0, second=0, microsecond=0) for ts in timestamps]

            df = pd.DataFrame({
                'Date': dates,
                'Open': opens,
                'High': highs,
                'Low': lows,
                'Close': closes,
                'Adj Close': adjclose_arr,
                'Volume': volumes,
                'Ticker': ticker
            })

            df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Adj Close'])

            return df

        elif resp.status_code == 429:
            logger.warning(f"HTTP 429 detectado para {ticker}. Esperando {espera}s antes de tentar novamente.")
            time.sleep(espera)
            espera *= 2  # espera exponencial
        else:
            logger.error(f"HTTP {resp.status_code} inesperado para {ticker}")
            break

    logger.error(f"Tentativas esgotadas para {ticker}")
    return None


# ============================================
# LISTAS DE TICKERS (MANTIDAS)
# ============================================

ibov_tickers = [
    "ALOS3.SA", "ABEV3.SA", "ASAI3.SA", "AURE3.SA", "AMOB3.SA", "AZUL4.SA", "AZZA3.SA", "B3SA3.SA",
    "BBSE3.SA", "BBDC3.SA", "BBDC4.SA", "BRAP4.SA", "BBAS3.SA", "BRKM5.SA", "BRAV3.SA", "BRFS3.SA",
    "BPAC11.SA", "CXSE3.SA", "CRFB3.SA", "CCRO3.SA", "CMIG4.SA", "COGN3.SA", "CPLE6.SA", "CSAN3.SA",
    "CPFE3.SA", "CMIN3.SA", "CVCB3.SA", "CYRE3.SA", "ELET3.SA", "ELET6.SA", "EMBR3.SA", "ENGI11.SA",
    "ENEV3.SA", "EGIE3.SA", "EQTL3.SA", "FLRY3.SA", "GGBR4.SA", "GOAU4.SA", "NTCO3.SA", "HAPV3.SA",
    "HYPE3.SA", "IGTI11.SA", "IRBR3.SA", "ISAE4.SA", "ITSA4.SA", "ITUB4.SA", "JBSS3.SA", "KLBN11.SA",
    "RENT3.SA", "LREN3.SA", "LWSA3.SA", "MGLU3.SA", "POMO4.SA", "MRFG3.SA", "BEEF3.SA", "MRVE3.SA",
    "MULT3.SA", "PCAR3.SA", "PETR3.SA", "PETR4.SA", "RECV3.SA", "PRIO3.SA", "PETZ3.SA", "PSSA3.SA",
    "RADL3.SA", "RAIZ4.SA", "RDOR3.SA", "RAIL3.SA", "SBSP3.SA", "SANB11.SA", "STBP3.SA", "SMTO3.SA",
    "CSNA3.SA", "SLCE3.SA", "SUZB3.SA", "TAEE11.SA", "VIVT3.SA", "TIMS3.SA", "TOTS3.SA", "UGPA3.SA",
    "USIM5.SA", "VALE3.SA", "VAMO3.SA", "VBBR3.SA", "VIVA3.SA", "WEGE3.SA", "YDUQ3.SA"
]

bdrs_tickers = [
    "JPMC34.SA", "BKNG34.SA", "BERK34.SA", "TSMC34.SA", "NFLX34.SA", "GOGL34.SA", "MELI34.SA", "M1TA34.SA",
    "RIGG34.SA", "CHVX34.SA", "INBR32.SA", "TSLA34.SA", "MSFT34.SA", "IVVB11.SA", "VISA34.SA", "BOAC34.SA",
    "GMCO34.SA", "COCA34.SA", "MSCD34.SA", "HOME34.SA", "AMGN34.SA", "AMZO34.SA", "JNJB34.SA", "SSFO34.SA",
    "NVDC34.SA", "AAPL34.SA", "CMCS34.SA", "VERZ34.SA", "UNHH34.SA", "DISB34.SA", "AIRB34.SA", "PFIZ34.SA",
    "AURA33.SA", "NIKE34.SA", "ITLC34.SA", "BABA34.SA"
]

commodities_tickers = [
    "BZ=F",  # Brent Crude Oil Futures (Futuros de Petróleo Brent)
    "GC=F",  # Gold Futures (Futuros de Ouro)
    "SI=F",  # Silver Futures (Futuros de Prata)
    "ZS=F",  # Soybean Futures (Futuros de Soja)
    "ZC=F",  # Corn Futures (Futuros de Milho)
    "KC=F",  # Coffee Futures (Futuros de Café)
    "LE=F"   # Live Cattle Futures (Futuros de Boi Gordo)
]
currency_tickers = ["USDBRL=X", "EURBRL=X"]

crypto_tickers = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
    "ADA-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "MATIC-USD", "LTC-USD",
    "LINK-USD", "SHIB-USD", "TON-USD", "NEAR-USD", "SUI-USD", "UNI-USD"
]

interest_tickers = [
    "^IRX",  # 13-Week Treasury Bill Yield (Taxa de retorno do T-Bill de 13 semanas, curto prazo EUA)
    "^FVX",  # 5-Year Treasury Yield (Taxa de retorno do título de 5 anos do Tesouro dos EUA)
    "^TNX",  # 10-Year Treasury Yield (Taxa de retorno do título de 10 anos do Tesouro dos EUA)
    "^TYX",  # 30-Year Treasury Yield (Taxa de retorno do título de 30 anos do Tesouro dos EUA)
    "EDV"    # Vanguard Extended Duration Treasury ETF (ETF que investe em Treasuries de longa duração)
]

categorias = {
    'acoes_ibov.csv': ibov_tickers,
    'bdrs.csv': bdrs_tickers,
    'commodities.csv': commodities_tickers,
    'moedas.csv': currency_tickers,
    'criptos.csv': crypto_tickers,
    'juros.csv': interest_tickers
}

# ============================================
# EXECUÇÃO PRINCIPAL
# ============================================
if __name__ == '__main__':
    garantir_pastas()

    hoje = datetime.now()
    dez_anos_atras = hoje - timedelta(days=365 * 10)

    for arquivo, tickers in categorias.items():
        caminho = os.path.join('dados', arquivo)
        if os.path.exists(caminho):
            df_exist = pd.read_csv(caminho, parse_dates=['Date'])
        else:
            df_exist = pd.DataFrame()

        novos = []
        logger.info(f"Iniciando coleta para {arquivo} ({len(tickers)} tickers)")

        for ticker in tqdm(tickers, desc=arquivo):
            try:
                df = obter_dados_historicos_yahoo(ticker, dez_anos_atras, hoje)
                if df is not None:
                    if not df_exist.empty and ticker in df_exist['Ticker'].unique():
                        ult_data = df_exist[df_exist['Ticker'] == ticker]['Date'].max()
                        df = df[df['Date'] > ult_data]
                    if not df.empty:
                        novos.append(df)
                        logger.info(f"Atualizados {len(df)} registros para {ticker}")
                else:
                    logger.error(f"Sem dados para {ticker}")
            except Exception as e:
                logger.exception(f"Erro coletando {ticker}: {e}")
            time.sleep(1)

            if novos:
                df_final = pd.concat([df_exist] + novos, ignore_index=True)
                df_final.drop_duplicates(['Date', 'Ticker'], inplace=True)
                df_final.sort_values(['Ticker', 'Date'], inplace=True)

                # REMOVE 'Volume' se for todo 0
                if 'Volume' in df_final.columns and (df_final['Volume'] == 0).all():
                    df_final = df_final.drop(columns=['Volume'])
                    logger.info(f"Coluna Volume removida de {arquivo} (todos zeros)")

                # 🚀 Salva CSV com delimitador ; e separador decimal ,
                df_final.to_csv(caminho, index=False, sep=';', decimal=',')

                logger.info(f"Arquivo salvo: {caminho} ({len(df_final)} registros)")
            else:
                logger.info(f"Nenhuma atualização para {arquivo}")


    logger.info("Coleta completa!")
