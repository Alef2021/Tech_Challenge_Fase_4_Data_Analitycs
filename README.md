# 🩺 Sistema Preditivo de Risco de Obesidade

## 🚀 Visão Geral do Projeto

Este projeto foi desenvolvido como parte do **Tech Challenge - Fase 4 - Data Analytics** da Pós-Graduação em Tecnologia e visa criar um **sistema preditivo** para auxiliar profissionais de saúde a diagnosticar e prever o risco de obesidade em pacientes.

O sistema utiliza técnicas de Machine Learning (ML) para analisar fatores comportamentais, históricos de habitos de uma pessoa, fornecendo um diagnóstico de risco em uma das 3 classes  definidas.

### 🔗 Link do Aplicativo Streamlit

Você pode acessar a aplicação preditiva aqui: **[https://aleftc4.streamlit.app/](https://aleftc4.streamlit.app/)**


### 🎯 Objetivos de Entrega

O projeto cumpriu os seguintes requisitos:
* ✅ **Pipeline de ML:** Demonstração completa de *feature engineering* e treinamento do modelo.
* ✅ **Assertividade:** Modelo com acurácia acima de 75% (Modelo XGBoost alcançou 88% na classificação em 3 classes).
* 🚧 **Deploy:** Modelo implantado em uma aplicação preditiva utilizando Streamlit (`app.py`).
* 🚧 **Visão Analítica:** Construção de um painel com principais *insights* (Dashboard a ser entregue separadamente).
* 🚧 **Documentação:** Arquivo de entrega contendo os links do App, Dashboard e Repositório (Requisito de documentação).



## ⚙️ Pipeline de Machine Learning

### 1. Feature Engineering e Pré-processamento

O pré-processamento dos dados foi crucial para preparar o conjunto para os modelos:

* **IMC (Índice de Massa Corporal):** Nova *feature* calculada a partir de `Peso` e `Altura`. As colunas **`Peso`** e **`Altura`** foram subsequentemente **removidas** das *features* de treinamento (`X`), pois o modelo se concentrou nas variáveis de hábito.
* **Discretização:** Colunas contínuas (`FCVC`, `NCP`, `CH2O`, `FAF`, `TUE`) que representavam frequência/níveis de consumo foram simplificadas para o primeiro dígito inteiro.
* **Conversão de Categóricas (Mapeamento de Hábitos):**
    * Variáveis binárias (`yes`/`no`) foram mapeadas para `1`/`0` (Ex: `Histórico_Familiar_Obesidade`, `Fumante`).
    * Variáveis ordinais de frequência (`no`, `Sometimes`, `Frequently`, `Always`) foram mapeadas para `0`, `1`, `2`, `3` (Ex: `Consumo_Alcool`, `Consumo_Alimento_Entre_Refeicoes`).
    * O `Meio_Transporte` foi classificado em 3 níveis de intensidade (`0`, `1`, `2`).
* **Tratamento da Variável Alvo:** A coluna original `Obesity_level` foi utilizada em três formatos para teste: **Binário**, **3 Classes** e **4 Classes**.

### 2. Treinamento e Avaliação de Modelos

Foram testados três algoritmos em três abordagens de classificação distintas: Regressão Logística, Random Forest e XGBoost.

| Abordagem | Classes | Modelo Vencedor | Acurácia |
| :---: | :---: | :---: | :---: |
| Binária | Não Obeso / Obeso | **XGBoost** | **91%** |
| **3 Classes** | **Normal / Sobrepeso / Obeso** | **XGBoost** | **88%** |
| 4 Classes | Abaixo do Peso / Normal / Sobrepeso / Obeso | XGBoost | 82% |

O modelo **XGBoost** com **3 Classes** (Normal/Sobrepeso/Obeso) foi o escolhido, oferecendo um bom nível de detalhe no diagnóstico com alta confiabilidade (88% de acurácia).

## 💻 Aplicação Preditiva (Streamlit)

A aplicação `app.py` permite que a equipe médica insira os dados do paciente e receba uma **previsão do Status de Risco** (Baixo, Médio, Alto) e um **nível de confiança**:

* **0:** `PESO NORMAL / BAIXO RISCO` / 🟢 Verde /
* **1:** `SOBREPESO / RISCO MÉDIO`  /🟠 Laranja /
* **2:** `OBESIDADE / ALTO RISCO`  /🔴 Vermelho /

