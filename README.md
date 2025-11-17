# 🧠 Sistema de Classificação de Risco de Morte por Insuficiência Cardíaca

Este projeto é um sistema completo que compara diferentes modelos de Machine Learning para prever o risco de óbito de pacientes com insuficiência cardíaca, utilizando o dataset **Heart Failure Clinical Records Dataset**, amplamente usado na área da saúde para pesquisas preditivas.

O programa foi desenvolvido como um **trabalho acadêmico**, incluindo:

- Tratamento e preparação dos dados  
- Treinamento de vários modelos de classificação  
- Cálculo das principais métricas de avaliação  
- Exibição dos resultados em uma interface gráfica  
- Guia de execução para qualquer computador  

A interface gráfica foi construída usando **Tkinter**, e os modelos foram treinados com **Scikit-Learn**.

---

## 📌 **Objetivo do Projeto**

Criar um sistema capaz de:

1. Carregar automaticamente o dataset clínico  
2. Treinar vários modelos de Machine Learning  
3. Comparar desempenho usando:  
   - Acurácia  
   - Recall  
   - Precisão  
   - F1-Score  
4. Exibir todos os resultados em uma tabela amigável em uma interface gráfica  

---

## 🗂 Tecnologias Utilizadas

- Python 3.10+  
- Tkinter (GUI)  
- Scikit-Learn  
- Pandas  
- StandardScaler  
- Modelos usados:
  - Regressão Logística  
  - Árvore de Decisão  
  - Random Forest  
  - KNN  
  - SVM  

---

## 📊 Dataset Utilizado

**Heart Failure Clinical Records Dataset**  
Link original no Kaggle:  
“andrewmvd/heart-failure-clinical-data”

O dataset contém 299 pacientes com diversas variáveis clínicas, incluindo idade, creatinina, diabetes, pressão sanguínea, entre outros, além da variável alvo **DEATH_EVENT**.

---

## 🖥 Como Executar o Projeto em Outro Computador

### ✅ **1. Instale o Python**
Baixe em: https://www.python.org/downloads/  
(Selecione *"Add Python to PATH"* durante a instalação)

---

### ✅ **2. Clone o repositório**

```bash
git clone https://github.com/projeto_ml.git
