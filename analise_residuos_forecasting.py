import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from scipy.stats import shapiro, skew, kurtosis
import statsmodels.api as sm
from loguru import logger

# Configuração do logger
os.makedirs('logs', exist_ok=True)
logger.add("logs/analise_residuos.log", level="INFO", rotation="10 MB", encoding="utf-8")

# Diretório dos resíduos
dir_residuos = 'logs/previsoes'
res_files = glob.glob(os.path.join(dir_residuos, 'residuos_*.csv'))

resultados = []

for file in tqdm(res_files, desc='Analisando resíduos'):
    try:
        df = pd.read_csv(file)
        if 'residuo' not in df.columns or 'previsto' not in df.columns:
            logger.warning(f'Arquivo {file} não possui colunas necessárias.')
            continue
        resid = df['residuo'].values
        nome = os.path.basename(file).replace('residuos_','').replace('.csv','')
        # Ljung-Box
        lb_p = acorr_ljungbox(resid, lags=[10], return_df=True)['lb_pvalue'].values[0]
        # Shapiro-Wilk
        shapiro_p = shapiro(resid)[1]
        # Breusch-Pagan (exog precisa de constante)
        exog = sm.add_constant(df['previsto'])
        bp_p = het_breuschpagan(resid, exog)[1]
        # Estatísticas básicas
        media = np.mean(resid)
        desvio = np.std(resid)
        assimetria = skew(resid)
        curtose = kurtosis(resid)
        resultados.append({
            'arquivo': nome,
            'ljungbox_p': lb_p,
            'shapiro_p': shapiro_p,
            'breuschpagan_p': bp_p,
            'media': media,
            'desvio': desvio,
            'skew': assimetria,
            'kurtosis': curtose
        })
    except Exception as e:
        logger.warning(f'Erro ao processar {file}: {e}')

# Salva análise de resíduos
df_res = pd.DataFrame(resultados)
df_res.to_csv('logs/analise_residuos.csv', index=False)

# Análise de desempenho e ranking
metricas_path = os.path.join(dir_residuos, 'metricas_forecasting.csv')
if os.path.exists(metricas_path):
    metricas = pd.read_csv(metricas_path)
    if 'arquivo' in metricas.columns and 'modelo' in metricas.columns:
        metricas['grupo'] = metricas['arquivo'].str.replace('.csv','')
        ranking = metricas.groupby(['grupo','modelo']).agg({'mae':'mean','mse':'mean'}).reset_index()
        ranking['rank_mae'] = ranking.groupby('grupo')['mae'].rank(method='min')
        ranking['rank_mse'] = ranking.groupby('grupo')['mse'].rank(method='min')
        ranking = ranking.sort_values(['grupo','rank_mae'])
        ranking.to_csv('logs/ranking_modelos.csv', index=False)
    else:
        ranking = pd.DataFrame()
else:
    ranking = pd.DataFrame()

# Resumo e interpretação
if not df_res.empty:
    logger.info("Resumo dos resíduos:")
    logger.info("\n" + str(df_res.describe()))
    
    # Interpretação dos testes
    logger.info("\nInterpretação dos testes estatísticos:")
    
    # Ljung-Box (autocorrelação)
    lb_rejeitados = (df_res['ljungbox_p'] < 0.05).sum()
    logger.info(f"Ljung-Box (autocorrelação): {lb_rejeitados} de {len(df_res)} séries apresentam autocorrelação significativa")
    
    # Shapiro-Wilk (normalidade)
    sw_rejeitados = (df_res['shapiro_p'] < 0.05).sum()
    logger.info(f"Shapiro-Wilk (normalidade): {sw_rejeitados} de {len(df_res)} séries não seguem distribuição normal")
    
    # Breusch-Pagan (heterocedasticidade)
    bp_rejeitados = (df_res['breuschpagan_p'] < 0.05).sum()
    logger.info(f"Breusch-Pagan (heterocedasticidade): {bp_rejeitados} de {len(df_res)} séries apresentam heterocedasticidade")
    
    # Análise dos resíduos por grupo
    logger.info("\nAnálise dos resíduos por grupo de ativos:")
    for grupo in ranking['grupo'].unique():
        grupo_res = df_res[df_res['arquivo'].str.contains(grupo)]
        logger.info(f"\nGrupo: {grupo}")
        logger.info(f"Média dos resíduos: {grupo_res['media'].mean():.4f}")
        logger.info(f"Desvio padrão: {grupo_res['desvio'].mean():.4f}")
        logger.info(f"Assimetria média: {grupo_res['skew'].mean():.4f}")
        logger.info(f"Curtose média: {grupo_res['kurtosis'].mean():.4f}")
else:
    logger.warning('Nenhum resíduo válido processado.')

if not ranking.empty:
    logger.info('\nModelos top-3 por grupo (menor MAE):')
    for grupo in ranking['grupo'].unique():
        logger.info(f'\nGrupo: {grupo}')
        top3 = ranking[ranking['grupo']==grupo][['modelo','mae','mse','rank_mae']].head(3)
        logger.info("\n" + str(top3))
else:
    logger.warning('\nRanking de modelos não disponível.') 