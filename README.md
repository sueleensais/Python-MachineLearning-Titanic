# Titanic: Machine Learning from Disaster

## Descrição
Este repositório contém meu estudo e desenvolvimento sobre o desafio clássico do Kaggle [Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic).
O objetivo é prever a sobrevivência dos passageiros do Titanic com base em variáveis demográficas e socioeconômicas, aplicando técnicas de Ciência de Dados e Machine Learning.

## Estrutura do Projeto
O projeto está dividido em duas partes principais:

**[Baseline](./titanic_baseline/)**
  
  Primeira versão do modelo, construída a partir de um tutorial do Kaggle.  
  - Objetivo: criar um modelo simples e funcional.  
  - Algoritmo: Random Forest.  

**[Advanced](./titanic_advanced/)** 

Versão expandida do projeto, estruturada como um pipeline completo de Data Science.
- Etapas: tratamento de valores ausentes, engenharia de features (Parte 1 e Parte 2), comparação de algoritmos, métricas avançadas e interpretabilidade.
- Algoritmos: Logistic Regression e XGBoost.
- Resultado: pontuação pública de 0.74641 no Kaggle.

## Objetivo
Demonstrar evolução prática em Ciência de Dados:
- Do aprendizado inicial (baseline)
- Até a construção de um case completo (avançado), aplicando boas práticas de modelagem, avaliação e comunicação de resultados.

## Tecnologias utilizadas
- Python (Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib, XGBoost) 
- Kaggle Notebook  
- GitHub (documentação e versionamento)

## Pontuação Pública (Kaggle)

| Versão     | Modelo           | Features principais                          | Pontuação Pública (Kaggle) |
|------------|-----------------|----------------------------------------------|----------------------------|
| **Baseline** | Random Forest   | Pclass, Sex, SibSp, Parch                   | **0.77511**                |
| **Advanced** | XGBoost         | Pclass, Sex, SibSp, Parch, FamilySize, Title | **0.74641**                |
 
Apesar do resultado inicial do modelo avançado ser inferior ao baseline, o pipeline trouxe maior robustez metodológica e abre espaço para otimizações futuras.

---

**Observação:** Este README apresenta a visão geral do repositório.
Cada pasta (titanic_baseline/ e titanic_advanced/) possui seu próprio README detalhando as etapas específicas de cada versão

