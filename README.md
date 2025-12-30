# 🧠 Tech Challenge — Fase 4 | Data Analytics  
## Sistema Preditivo de Risco de Obesidade

Projeto desenvolvido como parte do **Tech Challenge – Fase 4** da **Pós-Tech em Data Analytics (POSTECH)**, com o objetivo de aplicar conceitos de **Machine Learning, Análise Exploratória de Dados e Deploy de Modelos** em um cenário real de negócio na área da saúde.

### 🔗 Link do Aplicativo Streamlit

Você pode acessar a aplicação preditiva aqui: **[https://aleftc4versao3.streamlit.app/](https://aleftc4versao3.streamlit.app/)**

## 📌 Problema de Negócio

A obesidade é uma condição médica caracterizada pelo acúmulo excessivo de gordura corporal, podendo causar diversos riscos à saúde.  

Neste desafio, o objetivo foi **desenvolver um modelo de Machine Learning capaz de prever o risco de obesidade em pacientes**, auxiliando a equipe médica na **tomada de decisão clínica**, além de **construir uma visão analítica com insights relevantes sobre os fatores associados à obesidade**.

---

## 🎯 Objetivos do Projeto

- Desenvolver uma **pipeline completa de Machine Learning**
- Treinar um modelo com **assertividade superior a 75%**
- Criar uma **aplicação preditiva interativa utilizando Streamlit**
- Construir um **dashboard analítico com insights sobre obesidade**
- Apresentar os resultados de forma clara e orientada ao negócio

---

## 🗂️ Base de Dados

O projeto utiliza o dataset **`Obesity.csv`**, contendo informações demográficas, comportamentais e de estilo de vida.

### Principais variáveis:
- Gênero
- Idade
- Altura e Peso
- Histórico familiar de obesidade
- Consumo de alimentos calóricos
- Consumo de vegetais
- Consumo de água
- Frequência de atividade física
- Uso de tecnologia
- Consumo de álcool
- Meio de transporte  
- **Variável alvo:** Nível de Obesidade

---

## ⚙️ Pipeline de Machine Learning

1. Análise Exploratória de Dados (EDA)  
2. Limpeza e tratamento de dados  
3. Feature Engineering  
4. Codificação de variáveis categóricas  
5. Normalização e padronização  
6. Treinamento e avaliação de modelos  
7. Seleção do modelo final  
8. Salvamento do modelo  
9. Deploy em aplicação Streamlit  

---

## 🔍 Avaliação e Seleção do Modelo

Durante o desenvolvimento do projeto, foram testados diferentes algoritmos de Machine Learning, incluindo:

- Regressão Logística  
- Random Forest  
- XGBoost  

Os modelos foram comparados considerando métricas de desempenho, capacidade de generalização, estabilidade dos resultados e aderência ao contexto de negócio da área da saúde.

Após os testes, optou-se pela utilização do **XGBoost com classificação em três classes**, pois apresentou:

- Apresentou 85% de acurácia
- Melhor equilíbrio entre acurácia, precisão e recall  
- Maior capacidade de capturar relações não lineares  
- Melhor distinção entre níveis intermediários de risco  
- Resultados mais consistentes em validações  

### 📊 Estratégia de Classificação

O modelo final foi configurado para realizar uma **classificação multiclasse**, segmentando os pacientes em:

- 🟢 **Peso Normal / Baixo Risco**
- 🟡 **Sobrepeso / Risco Médio**
- 🔴 **Obesidade / Alto Risco**

Além da classe prevista, o modelo retorna a **probabilidade associada à predição**, aumentando a confiabilidade para apoio à decisão clínica.

---

## 🖥️ Aplicação Streamlit

### 1️⃣ Dashboard Analítico — Análise Exploratória
- Visualização interativa dos dados
- Filtros por gênero, idade, consumo de água e status de obesidade
- KPIs principais (IMC médio, idade média, etc.)
- Gráficos e insights comportamentais

### 2️⃣ Sistema Preditivo de Risco
- Inserção interativa dos dados do paciente
- Previsão do risco de obesidade em tempo real
- Probabilidade associada à previsão
- Identificação dos principais hábitos de risco e proteção
- Visualização do perfil comportamental do paciente

---

## 🗃️ Estrutura do Repositório

```
├── data/
│   └── Obesity.csv
├── notebooks/
│   ├── explocacao.ipynb
│   └── treinamento_teste_modelosML.ipynb
├── src/
│   ├── modelos/
│   │   └── modelo_obesidade_xgb_model_3_class.pkl
│   └── streamlit/
│       ├── app_explora.py        
│       └── pages/                
│           └── app.py            
├── gif_fundo/
│   └── giphy.gif
├── requirements.txt
└── README.md
```

---

## ▶️ Como Executar o Projeto

```bash
pip install -r requirements.txt
streamlit run app_explora.py
streamlit run app.py
```

---

## 🛠️ Tecnologias Utilizadas

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- XGBoost  
- Streamlit  

---

## 👨‍🎓 Autor

**Aluno:** Alef Souza Pereira  
**Curso:** Pós-Tech em Data Analytics  
**Instituição:** POSTECH  

---

## 📎 Considerações Finais

Este projeto demonstra a aplicação prática de Data Analytics e Machine Learning em um problema real da área da saúde, com foco em geração de insights e apoio à tomada de decisão clínica.
