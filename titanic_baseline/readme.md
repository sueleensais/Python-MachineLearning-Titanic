# **Titanic: Machine Learning from Disaster**

# **Descrição:**

O [Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) é um dos projetos e competições mais famosos e populares da plataforma Kaggle, projetado especificamente para iniciantes em Data Science e Machine Learning.

# **O Desafio:**

O desafio consiste em utilizar dados reais dos passageiros do Titanic (como idade, sexo, classe socioeconômica, tarifa paga, etc.) para construir um modelo preditivo que determine se um determinado passageiro sobreviveu ou não ao naufrágio ocorrido em 1912.

# **Dados**

**Fonte: Kaggle**

`train.csv`: Conterá os detalhes de um subconjunto dos passageiros a bordo (891, para ser exato) e, o que é mais importante, revelará se eles sobreviveram ou não, também conhecido como a "verdade fundamental".

`test.csv`: Conjunto de dados contém informações semelhantes, mas não revela a "verdade fundamental" para cada passageiro. Sua tarefa é prever esses resultados.

**Dicionário de Dados:**

→ Arquivo `train.csv`

- **Survived** → Sobreviveu (0 = não, 1 = sim)
- **Pclass** → Classe socioeconômica (1ª, 2ª ou 3ª classe)
- **PassengerId** → Identificador do passageiro
- **Name** → Nome completo do passageiro
- **Sex** → Sexo (male = masculino, female = feminino)
- **Age** → Idade em anos (valores fracionados possíveis, ex.: 34.5)
- **SibSp** → Número de irmãos/cônjuges a bordo
- **Parch** → Número de pais/filhos a bordo
- **Ticket** → Número do bilhete
- **Fare** → Tarifa (valor pago pela passagem)
- **Cabin** → Cabine (muitos valores ausentes)
- **Embarked** → Porto de embarque (C = Cherbourg, Q = Queenstown, S = Southampton)

→ Arquivo `test.csv` 

Mesma estrutura do `train.csv`, **exceto pela ausência da coluna `Survived`**, que é justamente o alvo a ser previsto.

# **Contexto histórico**

- **Data:** Na noite de 14 de abril de 1912, o Titanic colidiu com um iceberg no Atlântico Norte e afundou nas primeiras horas do dia 15.
- **Local:** Aproximadamente 640 km a leste da Ilha de Terra Nova, no Canadá.
- **Passageiros e tripulação:** Cerca de 2.224 pessoas estavam a bordo.
- **Mortes:** 1.502 pessoas perderam a vida, tornando-se uma das maiores tragédias marítimas da época.

# **Pipeline (Baseline)**
1. Entendimento do problema
2. Exploração inicial dos dados
3. Pré-processamento (baseline)
4. Modelagem
5. Avaliação

# **1. Entendimento do problema**

## **Importando as bibliotecas numpy e pandas:**

```python
import numpy as np
import pandas as pd
import os
```

## **Verificando os arquivos disponíveis:**

```python
for dirname, _, filenames in os.walk('/kaggle/input/competitions/titanic/'):
for filename in filenames:
print(os.path.join(dirname, filename))
```

## **Carregando os dados:**

```python
train_data = pd.read_csv('/kaggle/input/competitions/titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/competitions/titanic/test.csv')
```

# **2. Exploração inicial dos dados:**

## **Visualizando os dados** 

in:

```python
train_data.head()
```

out:

| **PassengerId** | **Survived** | **Pclass** | **Name** | **Sex** | **Age** | **SibSp** | **Parch** | **Ticket** | **Fare** | **Cabin** | **Embarked** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **0** | 1 | 0 | 3 | Braund, Mr. Owen Harris | male | 22.0 | 1 | 0 | A/5 21171 | 7.2500 | NaN |
| **1** | 2 | 1 | 1 | Cumings, Mrs. John Bradley (Florence Briggs Th... | female | 38.0 | 1 | 0 | PC 17599 | 71.2833 | C85 |
| **2** | 3 | 1 | 3 | Heikkinen, Miss. Laina | female | 26.0 | 0 | 0 | STON/O2. 3101282 | 7.9250 | NaN |
| **3** | 4 | 1 | 1 | Futrelle, Mrs. Jacques Heath (Lily May Peel) | female | 35.0 | 1 | 0 | 113803 | 53.1000 | C123 |
| **4** | 5 | 0 | 3 | Allen, Mr. William Henry | male | 35.0 | 0 | 0 | 373450 | 8.0500 | NaN |

in:

```python
test_data.head()
```

out:

|  | **PassengerId** | **Pclass** | **Name** | **Sex** | **Age** | **SibSp** | **Parch** | **Ticket** | **Fare** | **Cabin** | **Embarked** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **0** | 892 | 3 | Kelly, Mr. James | male | 34.5 | 0 | 0 | 330911 | 7.8292 | NaN | Q |
| **1** | 893 | 3 | Wilkes, Mrs. James (Ellen Needs) | female | 47.0 | 1 | 0 | 363272 | 7.0000 | NaN | S |
| **2** | 894 | 2 | Myles, Mr. Thomas Francis | male | 62.0 | 0 | 0 | 240276 | 9.6875 | NaN | Q |
| **3** | 895 | 3 | Wirz, Mr. Albert | male | 27.0 | 0 | 0 | 315154 | 8.6625 | NaN | S |
| **4** | 896 | 3 | Hirvonen, Mrs. Alexander (Helga E Lindqvist) | female | 22.0 | 1 | 1 | 3101298 | 12.2875 | NaN | S |

O método `.head()` do **pandas** mostra, por padrão, as **primeiras 5 linhas** do DataFrame. O dataset contém **891 linhas no `train.csv`** e **418 linhas no `test.csv`**.

## **Analisando a taxa de sobrevivência por gênero** 

in:

```python
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)
```

out:

```
% of women who survived: 0.7420382165605095
% of men who survived: 0.18890814558058924
```

O arquivo de amostra de submissão em *gender_submission.csv* pressupõe que todas as passageiras sobreviveram (e todos os passageiros do sexo masculino morreram). 
Com a análise inicial, verificamos que aproximadamente **74% das mulheres** sobreviveram, enquanto apenas **19% dos homens** conseguiram sobreviver. Esse resultado reflete a política de evacuação da época (“mulheres e crianças primeiro”), mostrando que o gênero foi um fator determinante. No entanto, essa análise se baseia em apenas uma coluna (`Sex`). 

# **3.Pré Processamento**

## **Importando a classe**

```python
from sklearn.ensemble import RandomForestClassifier
```

## **Definindo e selecionando as Features**
```python
y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
```

## **Transformando as variáveis categóricas**
```python
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])
```

# **4 - Modelagem**

## **Criando o primeiro modelo de Machine Learning**

```python
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
```
## **Treinando o modelo**
```python
model.fit(X, y)
```
## **Gerando as Previsões**
```python
predictions = model.predict(X_test)
```
# **Criando o Arquivo de Submissão**
```python
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
```

# **5. Avaliação**

Após treinar o primeiro modelo de Machine Learning (Random Forest), foi gerado o arquivo `submission.csv` e enviado para a competição **Titanic - Machine Learning from Disaster** no Kaggle.

- **Modelo**: Random Forest  
- **Features**: Pclass, Sex, SibSp, Parch  
- **Pontuação pública**: 0.77511, considerada um bom resultado para iniciantes e demonstrando capacidade de modelagem preditiva.

![Resultado da submissão](/result.png)

## **Nota sobre valores ausentes**

Neste primeiro modelo não foi realizado tratamento de valores ausentes.

A escolha se deve ao fato de que as variáveis utilizadas (`Pclass`, `Sex`, `SibSp`, `Parch`) não apresentam dados faltantes.

O objetivo foi construir um **baseline simples** e funcional.

Em versões futuras, serão aplicadas técnicas de imputação e engenharia de features para lidar com colunas como `Age`, `Cabin` e `Embarked`, que possuem valores ausentes e podem contribuir para melhorar a performance do modelo.



