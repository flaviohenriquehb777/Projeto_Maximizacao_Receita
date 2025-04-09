# Análise de Dados de Vendas para Otimização de Receita

## Visão Geral do Projeto

Este projeto teve como objetivo realizar uma análise exploratória e preditiva de uma base de dados de vendas (`df_vendas_EDA`) para identificar padrões, entender a relação entre preço, desconto e quantidade vendida, e, finalmente, determinar o preço de venda e o desconto ideais para maximizar a receita futura.

O processo envolveu as seguintes etapas:

1.  **Carregamento e Inspeção dos Dados:** A base de dados de vendas foi carregada e inspecionada para entender sua estrutura e colunas relevantes: 'PrecoVenda', 'PrecoOriginal', 'Desconto' e 'VendaQtd'.

2.  **Pré-processamento dos Dados:** As colunas numéricas foram escalonadas para preparar os dados para a modelagem. O `RobustScaler` foi utilizado para a coluna 'Desconto' (para mitigar a influência de outliers), e o `MinMaxScaler` foi aplicado a 'PrecoVenda', 'PrecoOriginal' e 'VendaQtd' para escalar os dados para um intervalo fixo.

3.  **Modelagem da Relação entre Preço e Quantidade:** Diversos algoritmos de regressão foram testados utilizando um pipeline do scikit-learn para prever a quantidade vendida com base no preço de venda, preço original e desconto. A Regressão Linear se destacou com um ajuste notavelmente bom aos dados.

4.  **Avaliação do Modelo:** O modelo de Regressão Linear foi avaliado utilizando métricas como RMSE e R² no conjunto de teste, e a curva de aprendizagem foi analisada para verificar a generalização e a ausência de overfitting significativo.

5.  **Otimização da Receita:** O modelo treinado foi utilizado para prever a quantidade vendida em diversos cenários de preço de venda e desconto. Mantendo o preço original em seu valor médio, iteramos sobre uma faixa de preços e descontos para identificar a combinação que maximiza a receita (Preço de Venda × Quantidade Prevista).

## Principais Insights e Resultados

A análise de otimização da receita sugeriu o seguinte cenário ideal:

* **Preço de Venda Ideal Estimado:** 19.92
* **Desconto Ideal Estimado:** 0.0 (Desconto Zero)
* **Melhor Receita Estimada:** 23254.72

Estes resultados indicam que, com base nos dados históricos e no modelo de regressão linear, a maior receita pode ser alcançada vendendo os produtos a um preço de aproximadamente 19.92 sem aplicar descontos (dentro do espaço de busca explorado).

## Tecnologias Utilizadas

* **Python:** Linguagem de programação principal para análise de dados e desenvolvimento do modelo.
* **Pandas:** Biblioteca para manipulação e análise de dados tabulares.
* **NumPy:** Biblioteca para computação numérica eficiente.
* **Scikit-learn (sklearn):** Biblioteca de machine learning utilizada para pré-processamento de dados (MinMaxScaler, RobustScaler), criação de pipelines, seleção de modelos (LinearRegression) e avaliação de modelos (train\_test\_split, cross\_val\_score, mean\_squared\_error, r2\_score, learning\_curve).
* **Matplotlib:** Biblioteca para visualização de dados, utilizada para gerar a curva de aprendizagem.
* **Seaborn:** Biblioteca de visualização de dados construída sobre o Matplotlib, utilizada para criar gráficos exploratórios (como boxplots, mencionados durante a análise).

## Autor

flaviohenriquehb777
