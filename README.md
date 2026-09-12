# 🔩 Classificação de Defeitos em Peças Industriais com CNN

Projeto educacional de **Machine Learning e Visão Computacional** para
classificar imagens de peças industriais em duas categorias:

-   `def_front` --- peça com defeito
-   `ok_front` --- peça em bom estado

A proposta do projeto é construir o processo aos poucos, em **Sprints**,
deixando cada etapa fácil de entender, testar e revisar. O notebook
parte da exploração do dataset, passa pela preparação e aumento das
imagens e chega ao treinamento e avaliação de uma **Rede Neural
Convolucional (CNN)**.

------------------------------------------------------------------------

## 🎯 Objetivo do Projeto

Criar um modelo capaz de observar a imagem frontal de uma peça
industrial e aprender padrões visuais que ajudem a responder:

> **Esta peça apresenta defeito ou está em bom estado?**

O projeto foi organizado pensando também em quem está começando na área
de dados. Por isso, cada Sprint representa uma etapa específica do
pipeline e pode ser acompanhada separadamente.

------------------------------------------------------------------------

## 🌿 Organização por Branches e Sprints

O desenvolvimento é separado em **branches**, permitindo acompanhar a
evolução do projeto sem concentrar todas as alterações de uma vez na
branch principal.

De forma simplificada, o fluxo é:

``` text
Dataset de imagens
        ↓
Exploração e conferência dos dados
        ↓
Preparação das imagens
        ↓
Divisão entre treino e validação
        ↓
Data Augmentation
        ↓
Construção da CNN
        ↓
Treinamento
        ↓
Avaliação dos resultados
```

Cada branch/Sprint registra uma etapa dessa evolução. Assim, quem
estiver estudando o projeto pode comparar as alterações realizadas ao
longo do desenvolvimento.

------------------------------------------------------------------------

## 📌 Sprint 1 --- Conhecendo e organizando os dados

Nesta primeira etapa, o objetivo é entender o dataset antes de construir
qualquer modelo.

São verificadas informações como:

-   quantidade de imagens;
-   classes existentes;
-   caminhos dos arquivos;
-   distribuição entre `def_front` e `ok_front`;
-   criação e organização das informações em um DataFrame com Pandas;
-   definição consistente dos rótulos das duas classes.

A ideia é simples: **antes de ensinar a máquina, precisamos entender os
dados que estamos entregando para ela.**

------------------------------------------------------------------------

## 🔎 Sprint 2 --- Análise exploratória

Nesta Sprint, o dataset é analisado com mais cuidado para identificar
possíveis problemas antes do treinamento.

O foco é conferir a qualidade e a distribuição dos dados, procurando
situações que poderiam prejudicar o modelo posteriormente.

Essa etapa ajuda a responder perguntas como:

> As duas classes estão representadas?

> Os dados parecem consistentes?

> Existe algum problema que deveria ser tratado antes de criar a CNN?

------------------------------------------------------------------------

## 🧹 Sprint 3 --- Preparação dos dados

Aqui o dataset começa a ser preparado para entrar no pipeline de Visão
Computacional.

O objetivo é garantir que as imagens e seus respectivos rótulos estejam
organizados de maneira consistente para as próximas etapas.

Essa preparação evita que problemas de estrutura do dataset sejam
confundidos posteriormente com problemas do modelo.

------------------------------------------------------------------------

## 🖼️ Sprint 4 --- Divisão dos dados e Data Augmentation

Nesta Sprint são preparadas as imagens que serão utilizadas pela CNN.

### Divisão estratificada

Os dados são separados em:

-   **Treino** --- imagens utilizadas para o modelo aprender;
-   **Validação** --- imagens utilizadas para verificar como o modelo se
    comporta com dados que não estão sendo usados para ajustar seus
    pesos.

A divisão é feita de forma **estratificada**, preservando
aproximadamente a proporção de `def_front` e `ok_front` nos dois
conjuntos.

Isso é importante porque uma validação contendo apenas uma das classes
produziria métricas enganosas.

### Data Augmentation

Também é aplicado **Data Augmentation** nas imagens de treinamento.

São utilizadas pequenas transformações, como:

-   rotação;
-   zoom;
-   alteração de brilho;
-   alteração de contraste;
-   espelhamento horizontal.

Essas transformações criam variações das imagens durante o treinamento
sem mudar sua classe.

Por exemplo:

``` text
Peça com defeito
      ↓
pequena rotação + zoom + mudança de brilho
      ↓
continua sendo uma peça com defeito
```

O objetivo é evitar que a CNN simplesmente memorize as imagens originais
e ajudá-la a aprender características mais gerais.

------------------------------------------------------------------------

## 🧠 Sprint 5 --- Arquitetura e treinamento da CNN

Nesta etapa é construída a **Rede Neural Convolucional (CNN)**.

De forma simplificada, a arquitetura segue este fluxo:

``` text
Imagem
  ↓
Data Augmentation
  ↓
Normalização
  ↓
Conv2D
  ↓
MaxPooling
  ↓
Conv2D
  ↓
MaxPooling
  ↓
Conv2D
  ↓
MaxPooling
  ↓
Flatten
  ↓
Dense
  ↓
Classificação binária
```

As camadas `Conv2D` procuram padrões visuais nas imagens, enquanto
`MaxPooling2D` reduz progressivamente a quantidade de informação
mantendo características relevantes.

Depois, `Flatten` transforma as informações extraídas em um formato que
pode ser utilizado pelas camadas densas.

Como existem apenas duas classes, a saída utiliza classificação binária:

``` text
def_front  ↔  ok_front
```

O modelo é treinado utilizando:

-   **Adam** como otimizador;
-   **Binary Crossentropy** como função de perda;
-   **Accuracy** como uma das métricas de acompanhamento.

Também é utilizado **EarlyStopping** para acompanhar o `val_loss`.
Assim, o número de épocas funciona como um limite máximo e o treinamento
pode ser interrompido quando a validação deixa de apresentar melhora
durante várias épocas.

------------------------------------------------------------------------

## 📊 Sprint 6 --- Avaliação do modelo

Depois do treinamento, não basta olhar apenas para a acurácia final.

Nesta Sprint são analisados diferentes indicadores para entender melhor
o comportamento da CNN.

### Curvas de treino e validação

São comparadas:

-   `accuracy`;
-   `val_accuracy`;
-   `loss`;
-   `val_loss`.

Essas curvas ajudam a visualizar se o modelo está realmente aprendendo e
também a procurar sinais de **overfitting**.

### Matriz de Confusão

A matriz de confusão mostra quantas imagens de cada classe foram
classificadas corretamente e onde ocorreram erros.

Exemplo conceitual:

``` text
                        PREVISTO
                   def_front   ok_front

REAL   def_front      acerto      erro
       ok_front       erro        acerto
```

Isso permite descobrir, por exemplo, se o modelo é muito bom em
reconhecer peças defeituosas, mas apresenta mais dificuldade com peças
em bom estado.

### Classification Report

Também são analisadas métricas como:

-   **Precision**
-   **Recall**
-   **F1-score**
-   **Support**

Dessa forma, a avaliação não depende apenas de uma única porcentagem de
acurácia.

------------------------------------------------------------------------

## ⚠️ Overfitting

Um dos pontos observados durante os experimentos é a diferença entre o
desempenho no treinamento e na validação.

De forma simples, **overfitting** acontece quando o modelo aprende muito
bem os exemplos utilizados durante o treinamento, mas não consegue
manter o mesmo desempenho em dados de validação.

Por isso são acompanhadas conjuntamente as curvas:

``` text
accuracy      ↑
val_accuracy  ↑

loss          ↓
val_loss      ↓
```

O comportamento das métricas ao longo das épocas é mais importante do
que analisar uma época isoladamente.

------------------------------------------------------------------------

## 🧪 Por que os resultados podem mudar entre execuções?

Mesmo utilizando o mesmo código, pequenas diferenças entre treinamentos
são esperadas.

Um dos motivos é o **Data Augmentation**, que pode apresentar ao modelo
variações diferentes das imagens em cada execução.

Além disso, existem outros elementos aleatórios no treinamento de redes
neurais, como inicialização dos pesos, embaralhamento dos dados e
Dropout.

Por isso, os resultados devem ser analisados como parte de um
experimento e não apenas como um único número final.

------------------------------------------------------------------------

## 🛠️ Tecnologias Utilizadas

-   **Linguagem:** Python
-   **Deep Learning:** TensorFlow / Keras
-   **Manipulação de dados:** Pandas / NumPy
-   **Processamento de imagens:** OpenCV
-   **Machine Learning e métricas:** scikit-learn
-   **Visualização:** Matplotlib / Seaborn
-   **Versionamento:** Git / GitHub
-   **Ambiente:** Jupyter Notebook

------------------------------------------------------------------------

## 📁 Estrutura conceitual do projeto

``` text
Projeto
│
├── Dataset
│   ├── def_front
│   └── ok_front
│
├── Notebook
│   ├── Sprint 1 — Organização dos dados
│   ├── Sprint 2 — Análise exploratória
│   ├── Sprint 3 — Preparação
│   ├── Sprint 4 — Split + Augmentation
│   ├── Sprint 5 — CNN + Treinamento
│   └── Sprint 6 — Avaliação
│
└── README.md
```

> A estrutura acima é uma representação didática do fluxo do projeto e
> não necessariamente reproduz literalmente os diretórios do
> repositório.

------------------------------------------------------------------------

## 🚀 O que este projeto demonstra?

Mais do que obter uma boa acurácia, este projeto busca demonstrar um
pipeline completo e compreensível:

**entender os dados → preparar → treinar → validar → avaliar.**

Para quem está começando em Machine Learning e Visão Computacional, isso
ajuda a perceber que construir um modelo não significa apenas chamar
`model.fit()`. A qualidade da divisão dos dados, o augmentation e a
interpretação das métricas são partes fundamentais do processo.

------------------------------------------------------------------------

## 📝 Licença

Este projeto foi desenvolvido para **fins educacionais**, como parte do
aprendizado de Machine Learning e Visão Computacional.

------------------------------------------------------------------------

## 👤 Autor

**Leonardo Badell**

🎓 Estudante de Machine Learning e Visão Computacional

------------------------------------------------------------------------

## ☕ Agradecimentos

Obrigado por visitar o projeto! A ideia deste repositório é registrar
não apenas o resultado final, mas também o processo de aprendizado, os
testes, os ajustes e as decisões tomadas em cada Sprint.

> *Em Machine Learning, preparar e entender bem os dados é tão
> importante quanto treinar o modelo.*

------------------------------------------------------------------------

⭐ **Se este projeto foi útil para seus estudos, considere deixar uma
estrela no repositório!**
