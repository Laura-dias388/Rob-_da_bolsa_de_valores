# 📈 Projeto Final - Programação para Ciência de Dados

## Objetivo

Desenvolver um algoritmo de Aprendizado de Máquina supervisionado para análise de ações da Bolsa de Valores B3 utilizando dados históricos da WEG (WEGE3).

---

## Tecnologias Utilizadas

- Python
- Pandas
- Scikit-Learn
- Yahoo Finance (yfinance)
- Streamlit
- Plotly

---

## Sobre a Empresa

A WEG S.A. é uma empresa multinacional brasileira que atua nos setores de equipamentos elétricos, automação industrial, geração, transmissão e distribuição de energia.

As ações da empresa são negociadas na B3 sob o ticker **WEGE3**.

---

## Metodologia

### 1. Coleta de Dados

Os dados históricos da ação WEGE3 foram obtidos utilizando a biblioteca **yfinance**, que acessa informações disponibilizadas pelo Yahoo Finance.

### 2. Criação da Variável Alvo

Foi criado um atributo chamado **Target**, utilizado como variável de saída do modelo.

- Target = 1 → quando o preço de fechamento é maior que o preço de abertura.
- Target = 0 → quando o preço de fechamento é menor ou igual ao preço de abertura.

### 3. Separação dos Dados

Os dados foram divididos em:

- Treinamento: anos de 2023 e 2024.
- Teste: ano de 2025.

### 4. Treinamento do Modelo

Foi utilizado o algoritmo **Decision Tree Classifier** da biblioteca Scikit-Learn para identificar padrões presentes nos dados históricos da ação.

### 5. Avaliação

O desempenho do modelo foi avaliado utilizando a métrica de acurácia, comparando as previsões realizadas com os resultados reais do período de teste.

### 6. Dashboard

Os resultados foram apresentados em um dashboard interativo desenvolvido com Streamlit.

---

## Estrutura do Projeto

```text
modulo4_IA/
│
├── analise.py
├── dashboard.py
├── requirements.txt
├── README.md
├── .gitignore
└── .venv/
```

---

## Instalação

Instalar as dependências:

```bash
pip install -r requirements.txt
```

---

## Execução

Executar a análise:

```bash
python analise.py
```

Executar o dashboard:

```bash
streamlit run dashboard.py
```

---

## Resultados

O modelo realiza previsões sobre a direção diária da ação WEGE3 utilizando uma Árvore de Decisão treinada com dados históricos.

Os resultados são apresentados através de métricas de desempenho, tabelas e gráficos interativos.

---

## Referências

- Yahoo Finance
- Scikit-Learn Documentation
- Pandas Documentation
- Streamlit Documentation
- Material da disciplina de Programação para Ciência de Dados

---

## Autor

Lauricea Vital De Toledo Dias

Projeto desenvolvido para a disciplina de Programação para Ciência de Dados.