import streamlit as st
import yfinance as yf
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

# ====================================

st.set_page_config(
    page_title="Projeto WEGE3",
    layout="wide"
)

st.title("Análise da WEGE3 com Machine Learning")

st.write("""
Projeto Final da disciplina de Programação para Ciência de Dados.

Modelo utilizado:
- Decision Tree Classifier

Ativo analisado:
- WEGE3
""")

# ====================================
# Filtro

ano_visualizacao = st.selectbox(
    "Ano para Visualização",
    [2023, 2024, 2025]
)

# ====================================
# Download de dados

dados = yf.download(
    "WEGE3.SA",
    start="2023-01-01",
    end="2026-01-01"
)

# ====================================
# Criar variável Target

dados["Target"] = (
    dados["Close"] > dados["Open"]
).astype(int)

# ====================================
# Filtrar ano escolhido

if ano_visualizacao == 2023:

    dados_filtrados = dados.loc[
        "2023-01-01":"2023-12-31"
    ]

elif ano_visualizacao == 2024:

    dados_filtrados = dados.loc[
        "2024-01-01":"2024-12-31"
    ]

else:

    dados_filtrados = dados.loc[
        "2025-01-01":"2025-12-31"
    ]

# ====================================
# Treinamento e teste 

treino = dados.loc[
    "2023-01-01":"2024-12-31"
]

teste = dados.loc[
    "2025-01-01":"2025-12-31"
]

# ====================================
# Preparação dos dados

X_treino = treino[
    ["Open", "High", "Low", "Volume"]
]

y_treino = treino["Target"]

X_teste = teste[
    ["Open", "High", "Low", "Volume"]
]

y_teste = teste["Target"]

# ====================================
# Modelo

modelo = DecisionTreeClassifier(
    random_state=42,
    max_depth=5
)

modelo.fit(
    X_treino,
    y_treino
)

# ====================================
# Previsões

previsoes = modelo.predict(
    X_teste
)

# ====================================
# Métricas

acuracia = accuracy_score(
    y_teste,
    previsoes
)

precisao = precision_score(
    y_teste,
    previsoes
)

f1 = f1_score(
    y_teste,
    previsoes
)

# ====================================
# Resultado

resultado = pd.DataFrame({
    "Real": y_teste,
    "Previsto": previsoes
})

acertos = (
    resultado["Real"] ==
    resultado["Previsto"]
).sum()

erros = (
    resultado["Real"] !=
    resultado["Previsto"]
).sum()

# ====================================
# Métricas na tela

st.subheader("Métricas do Modelo")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Acurácia",
    f"{acuracia:.2%}"
)

col2.metric(
    "Precisão",
    f"{precisao:.2%}"
)

col3.metric(
    "F1-Score",
    f"{f1:.2%}"
)

col4.metric(
    "Acertos",
    acertos
)

col5.metric(
    "Erros",
    erros
)

# ====================================
# Gráfico da ação

st.subheader(
    "Métricas do Modelo (Teste em 2025)"
)

st.line_chart(
    dados_filtrados["Close"]
)

# ====================================
# Distribuição 

st.subheader(
    "Distribuição da Variável Target"
)

contagem = (
    dados_filtrados["Target"]
    .value_counts() # conta quantas vezes cada valor aparece
)

st.bar_chart( # criação do grafico de barras
    contagem
)

percentuais = (
    dados_filtrados["Target"]
    .value_counts(normalize=True) # obtendo a proporção e multiplicando por 100 para obter a porcentagem
    * 100
)

st.write(
    f"Dias de queda (0): {contagem.get(0, 0)} ({percentuais.get(0, 0):.2f}%)"
)
# Mostra a distribuição das classes da variável Target, indicando quantos dias tiveram alta e quantos tiveram queda
st.write(
    f"Dias de alta (1): {contagem.get(1, 0)} ({percentuais.get(1, 0):.2f}%)"
)

# ====================================
# Tabela de previsões

st.subheader(
    "Primeiras Previsões"
)

resultado_reset = (
    resultado
    .reset_index()
)

st.dataframe(
    resultado_reset.head(20)
)

# ====================================
# Resumo

st.subheader(
    "Resumo dos Resultados"
)

st.write("""
O modelo foi treinado com dados de 2023 e 2024
e avaliado utilizando dados de 2025.
As métricas permanecem fixas independentemente
do ano selecionado para visualização.
""")