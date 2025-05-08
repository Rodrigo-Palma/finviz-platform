import requests
import pandas as pd
import os
import time
from datetime import datetime
from loguru import logger

# ─── CONFIGURAÇÃO ───────────────────────────────────────────────────────
PASTA_DADOS = "dados"
PASTA_LOGS = "logs"
CSV_HISTORICO = os.path.join(PASTA_DADOS, "historico_tesouro.csv")
URL = "https://www.tesourodireto.com.br/json/br/com/b3/tesourodireto/service/api/treasurybondsinfo.json"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json",
    "Referer": "https://www.tesourodireto.com.br/titulos/precos-e-taxas.htm"
}

os.makedirs(PASTA_DADOS, exist_ok=True)
os.makedirs(PASTA_LOGS, exist_ok=True)

logger.add(os.path.join(PASTA_LOGS, "execucao.log"), level="INFO", rotation="10 MB", encoding="utf-8")
logger.add(os.path.join(PASTA_LOGS, "erros.log"), level="ERROR", rotation="10 MB", encoding="utf-8")

# ─── FUNÇÕES ─────────────────────────────────────────────────────────────
def buscar_dados_tesouro(retries=3, delay=5):
    logger.info("Consultando dados do Tesouro Direto...")
    for tentativa in range(retries):
        try:
            response = requests.get(URL, headers=HEADERS, timeout=20)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.warning(f"Tentativa {tentativa + 1} falhou: {e}")
            time.sleep(delay)
    logger.error("Falha ao consultar dados do Tesouro após múltiplas tentativas.")
    raise Exception("Não foi possível buscar os dados do Tesouro.")

def parsear_dados(json_data, data_coleta):
    bonds = json_data.get("response", {}).get("TrsrBdTradgList", [])
    registros = []

    for bond_data in bonds:
        bond = bond_data.get("TrsrBd", {})
        registros.append({
            "Data_Coleta": data_coleta,
            "Titulo": bond.get("nm", ""),
            "Vencimento": datetime.strptime(bond.get("mtrtyDt", "").split("T")[0], "%Y-%m-%d").strftime("%d/%m/%Y") if bond.get("mtrtyDt") else None,
            "Taxa_Compra(%)": float(bond.get("anulInvstmtRate", 0)) if bond.get("anulInvstmtRate") else None,
            "Taxa_Venda(%)": float(bond.get("anulRedRate", 0)) if bond.get("anulRedRate") else None
        })

    return pd.DataFrame(registros)

def salvar_historico_incremental(df_novo):
    os.makedirs(PASTA_DADOS, exist_ok=True)

    # Garante que Data_Coleta esteja como datetime
    df_novo['Data_Coleta'] = pd.to_datetime(df_novo['Data_Coleta'])

    if os.path.exists(CSV_HISTORICO):
        df_antigo = pd.read_csv(CSV_HISTORICO, sep=";", parse_dates=["Data_Coleta"])

        registros_existentes = df_antigo[['Data_Coleta', 'Titulo']].drop_duplicates()
        df_novo_filtrado = df_novo.merge(
            registros_existentes,
            on=["Data_Coleta", "Titulo"],
            how="left",
            indicator=True
        ).query('_merge == "left_only"').drop(columns=["_merge"])

        if df_novo_filtrado.empty:
            logger.info("Nenhum novo registro para adicionar.")
            return

        df_final = pd.concat([df_antigo, df_novo_filtrado], ignore_index=True)
        df_final.drop_duplicates(subset=["Data_Coleta", "Titulo"], keep="last", inplace=True)
        df_final.sort_values(by=["Data_Coleta", "Titulo"], inplace=True)

    else:
        df_final = df_novo
        df_novo_filtrado = df_novo

    df_final.to_csv(CSV_HISTORICO, sep=";", index=False)
    logger.info(f"Histórico atualizado com {len(df_novo_filtrado)} novos registros.")

# ─── EXECUÇÃO PRINCIPAL ──────────────────────────────────────────────────
if __name__ == "__main__":
    data_coleta = datetime.now().strftime("%Y-%m-%d")
    try:
        dados_json = buscar_dados_tesouro()
        df_novo = parsear_dados(dados_json, data_coleta)

        if df_novo.empty:
            logger.warning("Nenhum dado encontrado na coleta de hoje.")
        else:
            salvar_historico_incremental(df_novo)
            logger.info("Coleta e salvamento concluídos com sucesso!")

    except Exception as e:
        logger.exception(f"Erro durante a execução: {e}")
