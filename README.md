# Aplicação Web – Maximização de Receita

[![Abrir a aplicação](docs/app_thumbnail.svg)](https://flaviohenriquehb777.github.io/Projeto_Maximizacao_Receita/)

Abra a aplicação acima. Abaixo, segue a documentação completa do projeto com o fluxo atualizado.

# Otimização de Receita com Machine Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Projeto de análise de dados de vendas e modelagem preditiva para otimização da receita, identificando a combinação ideal de preço de venda e desconto para maximizar lucros.**

## Sumário
- [Visão Geral do Projeto](#visão-geral-do-projeto)
- [Objetivos](#objetivos)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Base de Dados](#base-de-dados)
- [Metodologia de Análise e Modelagem](#metodologia-de-análise-e-modelagem)
- [Resultados Chave e Recomendações](#resultados-chave-e-recomendações)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Instalação e Uso](#instalação-e-uso)
- [Licença](#licença)
- [Contato](#contato)

## Visão Geral do Projeto:

Este projeto foca na aplicação de Machine Learning (Regressão Linear) para analisar dados de vendas e determinar as estratégias ideais de precificação e desconto que maximizem a receita. Através de uma análise exploratória detalhada e da construção de um modelo preditivo, buscamos fornecer insights acionáveis para otimização do desempenho de vendas.

## Objetivos:

* Realizar uma análise exploratória completa da base de dados de vendas.
* Entender a relação entre 'PrecoVenda', 'PrecoOriginal', 'Desconto' e 'VendaQtd'.
* Construir e avaliar um modelo de regressão capaz de prever a quantidade vendida.
* Identificar a combinação ideal de preço de venda e desconto para maximizar a receita.

## Estrutura do Projeto

- `dados/`: bases originais utilizadas nos estudos (não são necessárias para rodar a aplicação web; o treino usa `dataset_cafeterias_rj.xlsx`).
  - `dataset_cafeterias_rj.xlsx`
  - `dadosVenda.xlsx`
- `notebooks/`: cadernos Jupyter (EDA e estudos); não são obrigatórios para uso da aplicação.
  - `Projeto_Maximizacao_Receita_01.ipynb`
  - `Projeto_Maximizacao_Receita_EDA_Preprocess.ipynb`
  - `Projeto_Maximizacao_Receita_Final.ipynb`
- `src/`: código-fonte
  - `config/paths.py`: caminhos e constantes
  - `modeling/train_pipeline.py`: pipeline de treino, validação e geração de artefatos
  - `modeling/train_linear.py`: treino simples (compatível, mas o pipeline é preferível)
- `models/`: artefatos gerados pelo treino (ex.: `best_model_max_receita.pkl`, `curve_business_metric.csv`, `model_linear.json`, `shap_summary.png`)
- `docs/`: site estático consumindo `model_linear.json` e `curve_business_metric.csv`
  - `index.html`
  - `model_linear.json`
  - `curve_business_metric.csv`
- `.github/workflows/ci.yml`: CI para instalar, testar, treinar e publicar `docs/`
- `tests/`: testes unitários/integrados
- `requirements.txt`, `README.md`, `LICENSE.md`

## Base de Dados:

O projeto utiliza o arquivo `dataset_cafeterias_rj.xlsx` (constante `DADOS_AMOR_A_CAKES`), localizado na pasta `dados/`. Este dataset inclui as seguintes colunas principais:

* `preco_final`: Preço final de venda do produto.
* `preco_original`: Preço original do produto antes de qualquer desconto.
* `desconto_pct`: Desconto aplicado ao produto (0–0.04).
* `quantidade_vendida_dia` e `quantidade_vendida_mes`: Quantidades vendidas.

Com base nos testes de normalidade (Shapiro) e homogeneidade de variâncias (Levene), optamos por **não normalizar** (sem scaler) para a Regressão Linear e tratamos outliers com **winsorização por IQR** nas variáveis numéricas.

## Metodologia de Análise e Modelagem:

O desenvolvimento do projeto seguiu as seguintes etapas:

1.  **Carregamento e Inspeção de Dados:** Inicialização e compreensão da estrutura da base de dados.
2.  **Análise Exploratória de Dados (EDA):** Identificação de padrões, tendências e anomalias nas relações entre preço, desconto e quantidade vendida.
3.  **Pré-processamento:**
    * Tratamento de valores ausentes (se houver).
    * Escalonamento das variáveis numéricas para padronização.
    * Divisão dos dados em conjuntos de treino e teste.
4.  **Modelagem Preditiva:**
    * Avaliação de diversos algoritmos de regressão para prever a `VendaQtd`.
    * A **Regressão Linear** foi selecionada devido ao seu bom ajuste e interpretabilidade.
    * Utilização de *pipelines* do scikit-learn para encapsular etapas de pré-processamento e modelagem.
5.  **Avaliação do Modelo:**
    * Métricas como RMSE (Root Mean Squared Error) e R² (Coeficiente de Determinação) foram utilizadas para avaliar a performance do modelo.
    * Análise da curva de aprendizagem para verificar *underfitting* ou *overfitting*.
6.  **Otimização de Receita:**
    * A aplicação web varre descontos entre 0% e 4% (resolução configurável), calcula `preco_final = preco_original * (1 - desconto)` e prevê `quantidade_vendida`. Em seguida, maximiza `receita = preco_final * quantidade_vendida`.

## Resultados Chave e Recomendações:

Os resultados dependem dos coeficientes reais treinados no seu ambiente. A aplicação exibirá o desconto ideal, o preço final e a receita estimada com base no **modelo exportado**. Recomenda-se reavaliar periodicamente e monitorar a elasticidade de demanda em campanhas reais.

## Tecnologias Utilizadas:

* **Python:** Linguagem de programação principal.
* **Pandas:** Para manipulação e análise de dados tabulares.
* **NumPy:** Para operações numéricas de alto desempenho.
* **Scikit-learn (sklearn):** Para pré-processamento (MinMaxScaler, RobustScaler), modelagem (LinearRegression, Pipeline), divisão de dados (train_test_split) e avaliação de modelos (cross_val_score, mean_squared_error, r2_score, learning_curve).
* **Matplotlib:** Para criação de gráficos, especialmente a curva de aprendizagem.
* **Seaborn:** Para visualizações estatísticas e aprimoramento estético dos gráficos.

## Instalação e Uso

Para configurar e executar este projeto em seu ambiente local, siga as instruções abaixo:

1. **Pré-requisitos**
   - Python 3.12 (recomendado; CI usa 3.12)
   - `pip`
   - (Opcional) Jupyter Lab para explorar os notebooks

2.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/seu-usuario/Projeto_Maximizacao_Receita.git](https://github.com/seu-usuario/Projeto_Maximizacao_Receita.git)
    cd Projeto_Maximizacao_Receita
    ```
    *(Lembre-se de substituir `seu-usuario` pelo seu nome de usuário do GitHub.)*

3. **Instale as dependências**
    ```bash
    pip install -r requirements.txt
    ```

4. **(Opcional) Acesse e execute os notebooks**
    ```bash
    jupyter lab
    ```
    * Navegue até `notebooks/` e abra: `Projeto_Maximizacao_Receita_01.ipynb` → `Projeto_Maximizacao_Receita_EDA_Preprocess.ipynb`.

5. **Execute os testes e o pipeline de treino**
   ```bash
   # Defina PYTHONPATH para permitir imports dos testes
   export PYTHONPATH=$(pwd)   # Linux/macOS
   # No Windows PowerShell: $env:PYTHONPATH = (Get-Location).Path

   pytest -q
   python -m src.modeling.train_pipeline
   ```
   - Artefatos gerados em `models/`: `best_model_max_receita.pkl`, `curve_business_metric.csv`, `model_linear.json`, `shap_summary.png`.
   - Para a aplicação web, copie (ou use a CI) para `docs/`:
   ```bash
   cp models/model_linear.json docs/model_linear.json
   cp models/curve_business_metric.csv docs/curve_business_metric.csv
   ```

6. **Suba o site estático localmente**
   ```bash
   python -m http.server 8000
   # Abra http://localhost:8000/docs/
   ```

## Boas práticas e versão de arquivos

- Artefatos pesados e dados brutos não são versionados (ver `.gitignore`).
- Os notebooks são materiais de apoio; a aplicação e a CI usam o código em `src/`.
- A CI
  - Instala dependências
  - Executa `pytest` com `PYTHONPATH`
  - Treina com `python -m src.modeling.train_pipeline`
  - Copia artefatos para `docs/` e publica GitHub Pages

## Integração com DagsHub (MLflow)

Com credenciais configuradas (secrets), os treinos podem ser registrados no DagsHub via MLflow. Configure:

- `MLFLOW_EXPERIMENT_NAME`: nome do experimento
- `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` ou token

Em seguida, habilite no pipeline (ou CI) o log de parâmetros, métricas e artefatos (best model e explicabilidade).

## Licença:

Este projeto está licenciado sob a Licença MIT. Para mais detalhes, consulte o arquivo [LICENSE.md](LICENSE.md) na raiz do repositório.

## Contato:

Se tiver alguma dúvida, sugestão ou quiser colaborar, sinta-se à vontade para entrar em contato:
-   **Nome:** Flávio Henrique Barbosa
-   **LinkedIn:** [Flávio Henrique Barbosa | LinkedIn](https://www.linkedin.com/in/fl%C3%A1vio-henrique-barbosa-38465938)
-   **Email:** flaviohenriquehb777@outlook.com