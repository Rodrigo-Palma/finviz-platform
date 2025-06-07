import requests
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
import os

# ─── CONFIGURAÇÃO ───────────────────────────────────────────────────────────────
os.makedirs('dados', exist_ok=True)
os.makedirs('logs', exist_ok=True)
logger.add('logs/execucao.log', level='INFO', rotation='10 MB', encoding='utf-8', enqueue=True)
logger.add('logs/erros.log', level='ERROR', rotation='10 MB', encoding='utf-8', enqueue=True)

# Série do BCB (SGS):
# 4189 = Meta da Taxa Selic (diária)
# 4390 = Taxa DI – 1 dia (CDI Over)
series_bcb = {
    'selic_meta': 4189,
    'cdi_over': 4390
}

def obter_serie_bcb(codigo_serie: int, inicio: datetime, fim: datetime) -> pd.DataFrame:
    """
    Baixa série histórica do SGS (Banco Central) via JSON.
    """
    url = (
        f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo_serie}/dados?"
        f"formato=json&dataInicial={inicio.strftime('%d/%m/%Y')}"
        f"&dataFinal={fim.strftime('%d/%m/%Y')}"
    )
    resp = requests.get(url)
    if resp.status_code != 200:
        logger.error(f"BCB {codigo_serie} HTTP {resp.status_code}")
        return None

    data = resp.json()
    df = pd.DataFrame(data)
    df['data']  = pd.to_datetime(df['data'], dayfirst=True)
    df['valor'] = df['valor'].str.replace(',', '.').astype(float)
    df.rename(columns={'data':'Date','valor':'Rate'}, inplace=True)
    return df

# ─── RODANDO A COLETA ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    hoje = datetime.now()
    dez_anos = hoje - timedelta(days=365*10)

    for nome, serie in series_bcb.items():
        arquivo = f"dados/juros_brasil_{nome}.csv"
        # carrega existente para incremental
        if os.path.exists(arquivo):
            df_exist = pd.read_csv(arquivo, parse_dates=['Date'], sep=';', decimal=',')
        else:
            df_exist = pd.DataFrame()

        logger.info(f"Iniciando {nome} (BCB Séries {serie})")
        try:
            df = obter_serie_bcb(serie, dez_anos, hoje)
            if df is None or df.empty:
                raise ValueError("Nenhum dado retornado")
            # incremental
            if not df_exist.empty:
                data_max = df_exist['Date'].max()
                df = df[df['Date'] > data_max]
            if not df.empty:
                df_final = pd.concat([df_exist, df], ignore_index=True)
                df_final.drop_duplicates(['Date'], inplace=True)
                df_final.sort_values('Date', inplace=True)
                # 🚀 Salva CSV com delimitador ; e separador decimal ,
                df_final.to_csv(arquivo, index=False, sep=';', decimal=',')
                logger.info(f"Salvo {len(df)} novos registros em {arquivo}")
            else:
                logger.info(f"Nenhuma atualização para {nome}")
        except Exception as e:
            logger.exception(f"Erro coletando {nome}: {e}")
