import numpy as np
from typing import Dict, List, Tuple
import pandas as pd
from dataclasses import dataclass
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EstadoMarkov:
    nome: str
    frequencia: float
    duracao_media: float
    media: float
    std: float
    cv_mean: float
    cv_std: float
    r2: float
    interpretacao: str

class MarkovStatesImprover:
    def __init__(self, min_frequencia: float = 0.01, min_duracao: float = 1.5):
        self.min_frequencia = min_frequencia
        self.min_duracao = min_duracao
        self.scaler = StandardScaler()
        
    def ajustar_numero_estados(self, categoria: str, volatilidade: float) -> int:
        """Ajusta o número de estados baseado na categoria e volatilidade do ativo."""
        if categoria == "CRYPTO":
            return 2 if volatilidade > 0.5 else 3
        elif categoria == "CURRENCY":
            return 2
        elif categoria == "COMMODITY":
            return 2 if volatilidade > 0.3 else 3
        else:  # STOCK
            return 3 if volatilidade > 0.2 else 2
            
    def validar_estado(self, estado: Dict) -> bool:
        """Valida se um estado é válido baseado em critérios mínimos."""
        return (
            estado['frequencia'] >= self.min_frequencia and
            estado['duracao_media'] >= self.min_duracao and
            estado['std'] > 0 and
            estado['media'] != 0
        )
        
    def classificar_volatilidade(self, std: float, media: float) -> str:
        """Classifica a volatilidade baseado no coeficiente de variação."""
        cv = std / abs(media) if media != 0 else 0
        if cv > 0.5:
            return "ALTA_VOLATILIDADE"
        elif cv > 0.2:
            return "VOLATILIDADE_MEDIA"
        else:
            return "BAIXA_VOLATILIDADE"
            
    def consolidar_estados(self, estados: Dict[str, EstadoMarkov]) -> Dict[str, EstadoMarkov]:
        """Consolida estados similares para reduzir redundância."""
        estados_validos = {
            k: v for k, v in estados.items()
            if self.validar_estado(v.__dict__)
        }
        
        if len(estados_validos) < 2:
            return estados
            
        estados_consolidados = {}
        for nome, estado in estados_validos.items():
            similar = False
            for est_cons in estados_consolidados:
                if abs(estado.media - estados_consolidados[est_cons].media) < 0.1:
                    similar = True
                    break
            if not similar:
                estados_consolidados[nome] = estado
                
        return estados_consolidados
        
    def identificar_regime(self, estados: Dict[str, EstadoMarkov], cv_mean: float, cv_std: float) -> str:
        """Identifica o regime predominante baseado nos estados e volatilidade."""
        if cv_mean > 0.5 or cv_std > 0.3:
            return "ALTA_VOLATILIDADE"
            
        estados_validos = [v for v in estados.values() if self.validar_estado(v.__dict__)]
        if not estados_validos:
            return "LATERAL"
            
        tendencia = np.mean([v.media for v in estados_validos])
        
        if abs(tendencia) < 0.01:
            return "LATERAL_ESTRITO"
        elif tendencia > 0:
            return "TENDENCIA_ALTA"
        else:
            return "TENDENCIA_BAIXA"
            
    def melhorar_estados(self, df: pd.DataFrame, categoria: str) -> pd.DataFrame:
        """Aplica todas as melhorias nos estados Markov."""
        logger.info(f"Melhorando estados para categoria: {categoria}")
        
        # Calcula volatilidade geral
        volatilidade = df['cv_mean'].mean()
        
        # Ajusta número de estados
        n_estados = self.ajustar_numero_estados(categoria, volatilidade)
        logger.info(f"Número de estados ajustado para: {n_estados}")
        
        # Consolida estados
        estados_dict = {
            row['ticker']: EstadoMarkov(
                nome=row['ticker'],
                frequencia=row['direction_accuracy'],  # Usando direction_accuracy como frequência
                duracao_media=1.0,  # Valor padrão já que não temos essa informação
                media=row['r2'],  # Usando r2 como média
                std=row['rmse'],  # Usando rmse como desvio padrão
                cv_mean=row['cv_mean'],
                cv_std=row['cv_std'],
                r2=row['r2'],
                interpretacao=row['regime']
            )
            for _, row in df.iterrows()
        }
        
        estados_consolidados = self.consolidar_estados(estados_dict)
        logger.info(f"Estados consolidados: {len(estados_consolidados)}")
        
        # Identifica regime
        regime = self.identificar_regime(estados_consolidados, df['cv_mean'].mean(), df['cv_std'].mean())
        logger.info(f"Regime identificado: {regime}")
        
        # Atualiza DataFrame
        df_melhorado = pd.DataFrame([est.__dict__ for est in estados_consolidados.values()])
        df_melhorado['regime'] = regime
        
        return df_melhorado
        
    def processar_arquivo(self, arquivo: str) -> pd.DataFrame:
        """Processa um arquivo CSV com estados Markov e aplica melhorias."""
        logger.info(f"Processando arquivo: {arquivo}")
        
        if not os.path.exists(arquivo):
            logger.error(f"Arquivo {arquivo} não encontrado!")
            logger.info("Verificando arquivos disponíveis...")
            arquivos = [f for f in os.listdir('.') if f.endswith('.csv')]
            if arquivos:
                logger.info(f"Arquivos CSV encontrados: {arquivos}")
            return pd.DataFrame()
        
        df = pd.read_csv(arquivo)
        categorias = df['categoria'].unique()
        
        dfs_melhorados = []
        for categoria in categorias:
            df_cat = df[df['categoria'] == categoria].copy()
            df_melhorado = self.melhorar_estados(df_cat, categoria)
            dfs_melhorados.append(df_melhorado)
            
        return pd.concat(dfs_melhorados, ignore_index=True)

def main():
    improver = MarkovStatesImprover()
    
    # Arquivo de entrada
    arquivo_entrada = 'resultados/forecasting_markov/metricas/metricas_forecasting_markov.csv'
    
    # Processa arquivo
    df_melhorado = improver.processar_arquivo(arquivo_entrada)
    
    if not df_melhorado.empty:
        # Salva resultados no mesmo diretório
        arquivo_saida = arquivo_entrada.replace('.csv', '_melhorado.csv')
        df_melhorado.to_csv(arquivo_saida, index=False)
        logger.info(f"Análise concluída e resultados salvos em {arquivo_saida}")
    else:
        logger.error("Nenhum arquivo de entrada válido encontrado!")

if __name__ == "__main__":
    main() 