import pytest
import pandas as pd
import numpy as np
from improve_markov_states import MarkovStatesImprover, EstadoMarkov

@pytest.fixture
def improver():
    return MarkovStatesImprover()

@pytest.fixture
def sample_states():
    return {
        "ESTADO1": EstadoMarkov(
            nome="ESTADO1",
            frequencia=0.3,
            duracao_media=2.0,
            media=0.1,
            std=0.05,
            cv_mean=0.5,
            cv_std=0.2,
            r2=0.7,
            interpretacao="TENDENCIA_ALTA"
        ),
        "ESTADO2": EstadoMarkov(
            nome="ESTADO2",
            frequencia=0.2,
            duracao_media=1.8,
            media=-0.1,
            std=0.04,
            cv_mean=0.4,
            cv_std=0.15,
            r2=0.6,
            interpretacao="TENDENCIA_BAIXA"
        )
    }

def test_ajustar_numero_estados(improver):
    # Teste para CRYPTO
    assert improver.ajustar_numero_estados("CRYPTO", 0.6) == 2
    assert improver.ajustar_numero_estados("CRYPTO", 0.4) == 3
    
    # Teste para CURRENCY
    assert improver.ajustar_numero_estados("CURRENCY", 0.3) == 2
    
    # Teste para COMMODITY
    assert improver.ajustar_numero_estados("COMMODITY", 0.4) == 2
    assert improver.ajustar_numero_estados("COMMODITY", 0.2) == 3
    
    # Teste para STOCK
    assert improver.ajustar_numero_estados("STOCK", 0.3) == 3
    assert improver.ajustar_numero_estados("STOCK", 0.1) == 2

def test_validar_estado(improver):
    estado_valido = {
        'frequencia': 0.02,
        'duracao_media': 2.0,
        'std': 0.1,
        'media': 0.05
    }
    assert improver.validar_estado(estado_valido) == True
    
    estado_invalido = {
        'frequencia': 0.005,
        'duracao_media': 1.0,
        'std': 0,
        'media': 0
    }
    assert improver.validar_estado(estado_invalido) == False

def test_classificar_volatilidade(improver):
    assert improver.classificar_volatilidade(0.6, 1.0) == "ALTA_VOLATILIDADE"
    assert improver.classificar_volatilidade(0.3, 1.0) == "VOLATILIDADE_MEDIA"
    assert improver.classificar_volatilidade(0.1, 1.0) == "BAIXA_VOLATILIDADE"

def test_consolidar_estados(improver, sample_states):
    estados_consolidados = improver.consolidar_estados(sample_states)
    assert len(estados_consolidados) == 2  # Estados diferentes devem ser mantidos
    
    # Teste com estados similares
    estados_similares = {
        "ESTADO1": EstadoMarkov(
            nome="ESTADO1",
            frequencia=0.3,
            duracao_media=2.0,
            media=0.1,
            std=0.05,
            cv_mean=0.5,
            cv_std=0.2,
            r2=0.7,
            interpretacao="TENDENCIA_ALTA"
        ),
        "ESTADO2": EstadoMarkov(
            nome="ESTADO2",
            frequencia=0.2,
            duracao_media=1.8,
            media=0.11,  # Muito similar ao ESTADO1
            std=0.05,
            cv_mean=0.5,
            cv_std=0.2,
            r2=0.7,
            interpretacao="TENDENCIA_ALTA"
        )
    }
    estados_consolidados = improver.consolidar_estados(estados_similares)
    assert len(estados_consolidados) == 1  # Estados similares devem ser consolidados

def test_identificar_regime(improver, sample_states):
    # Teste para alta volatilidade
    assert improver.identificar_regime(sample_states, 0.6, 0.3) == "ALTA_VOLATILIDADE"
    
    # Teste para tendência alta
    estados_tendencia_alta = {
        "ESTADO1": EstadoMarkov(
            nome="ESTADO1",
            frequencia=0.3,
            duracao_media=2.0,
            media=0.1,
            std=0.05,
            cv_mean=0.3,
            cv_std=0.1,
            r2=0.7,
            interpretacao="TENDENCIA_ALTA"
        )
    }
    assert improver.identificar_regime(estados_tendencia_alta, 0.3, 0.1) == "TENDENCIA_ALTA"
    
    # Teste para lateral
    estados_lateral = {
        "ESTADO1": EstadoMarkov(
            nome="ESTADO1",
            frequencia=0.3,
            duracao_media=2.0,
            media=0.001,
            std=0.05,
            cv_mean=0.3,
            cv_std=0.1,
            r2=0.7,
            interpretacao="LATERAL"
        )
    }
    assert improver.identificar_regime(estados_lateral, 0.3, 0.1) == "LATERAL_ESTRITO"

def test_melhorar_estados(improver):
    # Cria DataFrame de teste
    df = pd.DataFrame({
        'estado': ['ESTADO1', 'ESTADO2'],
        'frequencia': [0.3, 0.2],
        'duracao_media': [2.0, 1.8],
        'media': [0.1, -0.1],
        'std': [0.05, 0.04],
        'cv_mean': [0.5, 0.4],
        'cv_std': [0.2, 0.15],
        'r2': [0.7, 0.6],
        'interpretacao': ['TENDENCIA_ALTA', 'TENDENCIA_BAIXA']
    })
    
    df_melhorado = improver.melhorar_estados(df, "STOCK")
    
    assert 'regime' in df_melhorado.columns
    assert len(df_melhorado) <= len(df)  # Pode ter menos estados após consolidação
    assert all(df_melhorado['frequencia'] >= improver.min_frequencia)
    assert all(df_melhorado['duracao_media'] >= improver.min_duracao)