import yfinance as yf
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ====================================
# Baixando dados direto do site 

dados = yf.download(
    "WEGE3.SA",
    start="2023-01-01",
    end="2026-01-01"
)

# ====================================
# Criando variaveis target

dados["Target"] = (
    dados["Close"] > dados["Open"]
).astype(int)

# ====================================
# Criando Treino e Teste

treino = dados.loc["2023-01-01":"2024-12-31"]
teste = dados.loc["2025-01-01":"2025-12-31"]

# ====================================
# Definir atributos (X) e alvo (y)

X_treino = treino[["Open", "High", "Low", "Volume"]]
y_treino = treino["Target"]

X_teste = teste[["Open", "High", "Low", "Volume"]]
y_teste = teste["Target"]

# ====================================
# Criar e treinar árvore

modelo = DecisionTreeClassifier(
    random_state=42,
    max_depth=5 # a árvore pode criar até 5 níveis de decisão para não ocorrer decoreba
)

modelo.fit(X_treino, y_treino) # aorendizagem do modelo usando os dados obtidos

# ====================================
# Fazer previsões

previsoes = modelo.predict(X_teste)

# ====================================
# Avaliar desempenho

acuracia = accuracy_score(
    y_teste,
    previsoes
)

print(f"Acurácia: {acuracia:.4f}")

# ====================================
# Métricas adicionais

from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

precisao = precision_score(
    y_teste,
    previsoes
)

f1 = f1_score(
    y_teste,
    previsoes
)

print(f"Precisão: {precisao:.4f}")
print(f"F1-Score: {f1:.4f}")

# ====================================
# Acertos e erros

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

print(f"Acertos: {acertos}")
print(f"Erros: {erros}")

# ====================================
# Mostrar algumas previsões

print("\nPrimeiras previsões:")
print(resultado.head())