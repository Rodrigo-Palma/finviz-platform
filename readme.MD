# 📊 Finviz Platform

Uma plataforma robusta de análise e previsão financeira utilizando dados do Yahoo Finance, machine learning e pipelines otimizados.

## 🚀 Visão Geral

Este projeto automatiza a coleta, limpeza e modelagem de dados financeiros com foco em ativos brasileiros e internacionais:

- 📈 Ações do IBOVESPA e BDRs
- 💱 Moedas e commodities
- 🔐 Criptoativos e juros

Inclui modelos preditivos com:

- 🧠 XGBoost, LightGBM, CatBoost, Random Forest
- 🔎 Seleção de variáveis com SHAP
- 🧪 Otimização via Optuna
- 📅 Avaliação por TimeSeriesSplit

## 📦 Estrutura do Projeto

```bash
finviz-platform/
├── coletor.py                # Coleta automática de dados do Yahoo Finance
├── fazer_previsao.py         # Geração de previsões com modelos treinados
├── dados/                    # Dados brutos coletados
├── logs/                     # Logs de execução
├── requisitos.txt            # Dependências do projeto
├── .gitignore                # Arquivos ignorados pelo git
└── README.md
```

## 🛠️ Tecnologias
- Python 3.10+
- Pandas, Requests, TQDM
- Scikit-learn, XGBoost, SHAP, Optuna
- Loguru para logging

## 🧪 Executando localmente

```bash
pip install -r requisitos.txt
python coletor.py
python fazer_previsao.py
```

## 📘 Licença
Este projeto está sob a licença MIT.

---

> Desenvolvido por Rodrigo Palma, Giovani Cancherini e Luis Gabriel
