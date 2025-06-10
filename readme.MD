# 📊 TradeViz Platform

Uma plataforma robusta de análise e previsão financeira utilizando dados do Yahoo Finance, machine learning e pipelines otimizados.

## 🔮 Link do Projeto Live

[https://l1nq.com/HOQiU](https://l1nq.com/HOQiU)

## 🚀 Visão Geral

Este projeto automatiza a coleta, limpeza e modelagem de dados financeiros com foco em ativos brasileiros e internacionais:

- 📈 Ações do IBOVESPA e BDRs
- 💱 Moedas e commodities
- 🔐 Criptoativos e Tesouro Americano

Inclui modelos preditivos com:

- 🧠 XGBoost, LightGBM, CatBoost, Random Forest, LSTM, XGBoostRegressor, Markov Chains, Hidden Markov Models, ARIMA, Prophet
- 🔎 Seleção de variáveis com SHAP
- 🧪 Otimização via Optuna
- 📅 Avaliação por TimeSeriesSplit e validação walk-forward

## 📦 Estrutura do Projeto

```bash
finviz-platform/
├── coletor.py                        # Coleta automática de dados do Yahoo Finance
├── fazer_previsao.py                 # Geração de previsões com modelos treinados
├── ml_pipeline_forecasting.py        # Pipeline ARIMA/Prophet (forecasting tradicional)
├── ml_pipeline_forecasting_deep.py   # Pipeline Deep/ML (Random Forest, XGBoost, LSTM)
├── ml_pipeline_forecasting_markov.py # Pipeline Markov/Hidden Markov
├── ml_pipeline_financeiro.py         # Pipeline de classificação financeira (ensemble)
├── dados/                            # Dados brutos coletados
├── dados_transformados/              # Dados prontos para modelagem
├── resultados/
│   ├── forecasting_arima_prophet/    # Resultados do pipeline ARIMA/Prophet
│   ├── forecasting_deep/             # Resultados do pipeline Deep/ML
│   ├── forecasting_markov/           # Resultados do pipeline Markov/HMM
│   └── financeiro/                   # Resultados do pipeline de classificação
├── modelos_forecasting/              # Modelos de forecasting salvos
├── logs/                             # Logs de execução
├── requisitos.txt                    # Dependências do projeto
├── .gitignore                        # Arquivos ignorados pelo git
├── LICENSE                           # Licença Apache 2.0
└── README.md
```

Cada subdiretório em `resultados/` contém:
- `plots/` — Gráficos gerados
- `residuos/` — CSVs de resíduos dos modelos
- `previsoes/` — CSVs de previsões
- `metricas/` — CSVs de métricas de avaliação
- `analises/` — (Markov) Análises de estados HMM
- `avaliacao/`, `trials/`, `modelos/` — (Financeiro) Avaliações, trials do Optuna e modelos salvos

## 🛠️ Tecnologias
- Python 3.10+
- Pandas, Requests, TQDM
- Scikit-learn, XGBoost, LightGBM, CatBoost, SHAP, Optuna
- Loguru para logging
- TensorFlow/Keras para LSTM
- hmmlearn para Hidden Markov Models
- Matplotlib, Seaborn para visualização

## 🧪 Executando localmente

```bash
pip install -r requisitos.txt
python coletor.py
python cdi-selic.py
python tesouro.py
python engenharia.py
python feature_selection.py
python ml_otimizador_janela
python ml_pipeline_forecasting.py
python ml_pipeline_forecasting_deep.py
python ml_pipeline_forecasting_markov.py
python ml_pipeline_financeiro.py
```

## 📂 Organização dos Resultados

- Todos os resultados, gráficos, resíduos, previsões e métricas são salvos em subdiretórios dentro de `resultados/`, separados por tipo de modelo.
- Os logs de execução ficam em `logs/`.
- Os modelos salvos ficam em `modelos_forecasting/` ou `resultados/financeiro/modelos/`.

## 📘 Licença

Este projeto está licenciado sob a [Apache License 2.0](LICENSE).

Você pode usar, modificar, distribuir e até mesmo utilizar comercialmente este software, desde que mantenha o aviso de copyright e a licença.
A licença também oferece proteção contra disputas de patente e exige que modificações significativas sejam documentadas.

---

> Desenvolvido por Rodrigo Palma, Giovani Cancherini, Luis Marquetti

