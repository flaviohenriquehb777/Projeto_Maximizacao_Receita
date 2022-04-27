# Otimização de Receita com Machine Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)

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

## Estrutura do Projeto:

O repositório está organizado para facilitar a navegação e compreensão:

-   `dados/`: Contém a base de dados original do projeto.
    -   `dadosVenda.xlsx`: A base de dados principal utilizada para a análise e modelagem.
-   `img/`: Armazena os gráficos e visualizações gerados durante a análise.
    -   `curva_aprendizagem_regressao_linear.png`: Gráfico da curva de aprendizagem do modelo de regressão linear.
-   `notebooks/`: Contém os notebooks Jupyter que detalham o processo do projeto.
    -   `Projeto_Maximizacao_Receita_01.ipynb`: Notebook inicial com a análise exploratória, pré-processamento e os primeiros passos da modelagem.
    -   `Projeto_Maximizacao_Receita_Final.ipynb`: Notebook final com a implementação completa do modelo, avaliação, otimização da receita e conclusões.
-   `README.md`: Este arquivo, que fornece uma visão geral detalhada do projeto.
-   `LICENSE.md`: Arquivo contendo os termos da licença do projeto (MIT).
-   `requirements.txt`: Lista de todas as bibliotecas Python e suas versões necessárias para executar o projeto.

## Base de Dados:

O projeto utiliza o arquivo `dadosVenda.xlsx`, localizado na pasta `dados/`. Este dataset inclui as seguintes colunas principais:

* `PrecoVenda`: Preço final de venda do produto.
* `PrecoOriginal`: Preço original do produto antes de qualquer desconto.
* `Desconto`: Desconto aplicado ao produto.
* `VendaQtd`: Quantidade de produtos vendidos.

A base foi sujeita a etapas de pré-processamento, incluindo o escalonamento das colunas numéricas usando `RobustScaler` (para 'Desconto') e `MinMaxScaler` (para 'PrecoVenda', 'PrecoOriginal' e 'VendaQtd') para otimizar a performance do modelo.

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
    * Com o modelo treinado, foi implementada uma função para simular diferentes cenários de preço e desconto.
    * O objetivo foi encontrar a combinação que resulta na `VendaQtd` prevista mais alta, levando à maximização da receita (`PrecoVenda * VendaQtd`).

## Resultados Chave e Recomendações:

A análise de otimização da receita apontou para o seguinte cenário ideal, com base no modelo de Regressão Linear:

* **Preço de Venda Ideal Estimado:** **R$ 19.92**
* **Desconto Ideal Estimado:** **0.0% (Desconto Zero)**
* **Melhor Receita Estimada:** **R$ 23.254,72**

Estes resultados sugerem que, dentro do escopo dos dados analisados e das premissas do modelo, a receita máxima é alcançada com um preço de venda específico e sem a aplicação de descontos adicionais. É crucial monitorar a implementação dessas recomendações e reavaliar o modelo periodicamente com novos dados de vendas.

Nota: após as melhorias recentes no pipeline e nos experimentos (uso do dataset real no CI, avaliação de múltiplos modelos), os valores acima podem variar. Para confirmar o melhor modelo e atualizar as recomendações, consulte o resumo do último run do GitHub Actions ou os artifacts gerados (por exemplo, `artifacts/metrics_best.json`).

<!-- MODEL_RESULTS_START -->

### Resumo automático do último run

- Treino
  - RMSE: N/A
  - R2: N/A
- Melhor Modelo
  - Modelo: N/A
  - RMSE: N/A
  - R2: N/A

<!-- MODEL_RESULTS_END -->

## Tecnologias Utilizadas:

* **Python:** Linguagem de programação principal.
* **Pandas:** Para manipulação e análise de dados tabulares.
* **NumPy:** Para operações numéricas de alto desempenho.
* **Scikit-learn (sklearn):** Para pré-processamento (MinMaxScaler, RobustScaler), modelagem (LinearRegression, Pipeline), divisão de dados (train_test_split) e avaliação de modelos (cross_val_score, mean_squared_error, r2_score, learning_curve).
* **Matplotlib:** Para criação de gráficos, especialmente a curva de aprendizagem.
* **Seaborn:** Para visualizações estatísticas e aprimoramento estético dos gráficos.

## Instalação e Uso:

Para configurar e executar este projeto em seu ambiente local, siga as instruções abaixo:

1.  **Pré-requisitos:**
    * Python 3.8+
    * `pip` (gerenciador de pacotes do Python)
    * Jupyter Lab ou Jupyter Notebook

2.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/seu-usuario/Projeto_Maximizacao_Receita.git](https://github.com/seu-usuario/Projeto_Maximizacao_Receita.git)
    cd Projeto_Maximizacao_Receita
    ```
    *(Lembre-se de substituir `seu-usuario` pelo seu nome de usuário do GitHub.)*

3.  **Crie o arquivo `requirements.txt`:**
    * Certifique-se de que está na raiz do projeto.
    * **No PowerShell (Windows):**
        ```powershell
        pip freeze | Out-File -FilePath requirements.txt -Encoding UTF8
        ```
    * **No Linux/macOS (ou Git Bash no Windows):**
        ```bash
        pip freeze > requirements.txt
        ```
    *(**Importante:** Faça isso *depois* de ter todas as bibliotecas usadas nos notebooks instaladas no seu ambiente Python.)*

4.  **Instale as dependências:**
    * Com o `requirements.txt` criado, instale todas as bibliotecas necessárias:
        ```bash
        pip install -r requirements.txt
        ```

5.  **Acesse e Execute os Notebooks:**
    * Inicie o Jupyter Lab na raiz do projeto:
        ```bash
        jupyter lab
        ```
    * Navegue até a pasta `notebooks/` e abra os notebooks na sequência (`Projeto_Maximizacao_Receita_01.ipynb` e `Projeto_Maximizacao_Receita_Final.ipynb`) para reproduzir a análise e os resultados.

## MLOps e CI/CD (Amor a Cakes)

Este repositório foi atualizado para um fluxo MLOps profissional com foco em produção, permitindo versionamento de dados, reprodutibilidade, rastreamento de experimentos e automação via CI/CD.

- Tecnologias adicionadas:
  - `DVC` para versionamento de dados e pipelines
  - `MLflow` para tracking de experimentos e artefatos
  - `DagsHub` para colaboração (espelhamento Git, DVC e MLflow)
  - GitHub Actions para CI/CD

### Estrutura adicional

- `src/train.py`: script de treino que replica a lógica dos notebooks (escalonamento + regressão linear), registra métricas no MLflow e salva artefatos.
- `dvc.yaml`: pipeline DVC com um estágio `train` dependente do dataset e do script de treino.
- `models/model_linear.joblib`: modelo treinado salvo via `joblib`.
- `artifacts/metrics.json`: métricas de avaliação (`rmse`, `r2`).
- `.github/workflows/mlops.yml`: workflow CI que instala dependências e executa o treino.
- `.env.example`: variáveis de ambiente para integração com DagsHub/MLflow.

### Como rodar o treino localmente

1. Instale dependências: `pip install -r requirements.txt`
2. Opcional: configure o MLflow remoto (DagsHub) criando um `.env` com base no `.env.example`.
3. Execute: `python src/train.py`
4. Resultados:
   - Modelo: `models/model_linear.joblib`
   - Métricas: `artifacts/metrics.json`
   - Tracking MLflow: pasta `mlruns/` local ou servidor remoto, se configurado.

### Pipeline DVC

- Adicionar/atualizar dados com DVC:
  - Se o arquivo de dados está em Git, primeiro remova do Git: `git rm -r --cached dados/dadosVenda.xlsx`
  - Adicione com DVC: `dvc add dados/dadosVenda.xlsx`
- Executar pipeline: `dvc repro` (roda `python src/train.py` e atualiza `outs`)
- Para compartilhar dados com a equipe: configure `dvc remote` no DagsHub.

Exemplo de configuração de remoto (ajuste `OWNER/REPO`):

```bash
dvc remote add -d dagshub https://dagshub.com/OWNER/REPO.dvc
dvc remote modify dagshub user "$DAGSHUB_OWNER"
dvc remote modify dagshub password "$DAGSHUB_TOKEN"
```

### MLflow e DagsHub

- Local por padrão: `mlruns` na raiz do projeto.
- Para usar DagsHub como servidor MLflow:
  - Crie o repositório em `https://dagshub.com/flaviohenriquehb777`
  - Sete `DAGSHUB_OWNER`, `DAGSHUB_REPO` e `DAGSHUB_TOKEN` como variáveis de ambiente.
  - Opcionalmente defina `MLFLOW_TRACKING_URI=https://dagshub.com/${DAGSHUB_OWNER}/${DAGSHUB_REPO}.mlflow`
  - O script `src/train.py` detecta e configura o tracking automaticamente.

### CI/CD (GitHub Actions)

[![CI Status](https://github.com/flaviohenriquehb777/Projeto_Maximizacao_Receita/actions/workflows/mlops.yml/badge.svg?branch=main)](https://github.com/flaviohenriquehb777/Projeto_Maximizacao_Receita/actions/workflows/mlops.yml)

- Workflow: `.github/workflows/mlops.yml`
- Em cada push/PR nas branches `main`/`master`:
  - Matrix de Python (`3.10`, `3.11`) com cache de `pip`
  - Cache de DVC baseado em `dvc.lock`
  - Executa `pre-commit` (Black, isort, Flake8, trailing whitespace, EOF)
  - Executa testes com cobertura (`pytest -q --cov=src --cov-report=xml --cov-report=html`)
  - Upload de cobertura (`coverage.xml`) e HTML (`htmlcov/`) como artifacts
  - Treino e experimentos com fallback sintético quando o dataset não está no remoto
  - Publica modelos e métricas como artifacts
- Para usar DagsHub no CI, configure os `secrets`:
  - `DAGSHUB_OWNER`, `DAGSHUB_REPO`, `DAGSHUB_TOKEN` e opcional `MLFLOW_TRACKING_URI`

### Fluxo Git recomendado

- Branches: `main` (produção) e `feature/*` (desenvolvimento)
- Pull Requests com revisão
- Commits pequenos e descritivos
- Para dados: use `dvc add` e `dvc push` para sincronizar com o remoto

### Testes em notebooks

- Notebooks permanecem funcionais. Utilize-os para EDA e validação local.
- Para experimentos reproduzíveis, prefira `src/train.py` com MLflow.

## Licença:

Este projeto está licenciado sob a Licença MIT. Para mais detalhes, consulte o arquivo [LICENSE.md](LICENSE.md) na raiz do repositório.

 

## Estrutura atualizada (refatoração)

Para uma organização profissional e maior qualidade do código, o projeto foi estruturado como um pacote Python dentro de `src/`:

- `src/projeto_maximizacao_receita/`: pacote com módulos por domínio
  - `config/paths.py`: caminhos e constantes (`PROJECT_ROOT`, `DADOS_VENDA_PATH`, alias `DADOS_AMOR_A_CAKES`)
  - `utils/`: utilitários gerais
  - `ml/`: utilitários de machine learning
  - `viz/`: visualizações
  - `stats/`: rotinas estatísticas
- `src/train.py` e `src/experiments.py`: entrypoints que usam o pacote (mantêm compatibilidade com DVC e testes)
- `.pre-commit-config.yaml`: hooks consolidados (`ruff`, `ruff-format`, `black`) e exclusões para `.github/workflows/*.yml` em whitespace/EOL
- `.gitignore`: ignora saídas de `models/` e `artifacts/` gerenciadas via DVC

Importante: há um módulo de compatibilidade em `src/config/paths.py` que reexporta os símbolos principais do novo pacote para não quebrar notebooks e testes legados.
### Como evitar erros no Actions

O CI foi endurecido para seguir mesmo sem dataset, usando dados sintéticos. Para utilizar o dataset real e não ver avisos do DVC:

1. Configure o remoto DVC no seu ambiente local (ajuste `OWNER/REPO`):

```bash
dvc remote add -d dagshub https://dagshub.com/OWNER/REPO.dvc
dvc remote modify dagshub auth basic
dvc remote modify dagshub user "$DAGSHUB_OWNER"
dvc remote modify dagshub password "$DAGSHUB_TOKEN"
```

2. Envie o dataset para o remoto:

```bash
dvc push dados/dadosVenda.xlsx.dvc
```

3. No GitHub, adicione os segredos em Settings → Secrets and variables → Actions:

- `DAGSHUB_OWNER`, `DAGSHUB_REPO`, `DAGSHUB_TOKEN` (ou `DAGSHUB_PASSWORD`)
- Opcional: `MLFLOW_TRACKING_URI` se quiser usar o servidor MLflow do DagsHub

Com isso, o passo `dvc pull` baixa o dataset no CI e o treino usa dados reais. Sem dados no remoto, o CI continua rodando com fallback sintético, sem falhar.

## Contato:

Se tiver alguma dúvida, sugestão ou quiser colaborar, sinta-se à vontade para entrar em contato:
- **Nome:** Flávio Henrique Barbosa
- **LinkedIn:** [Flávio Henrique Barbosa | LinkedIn](https://www.linkedin.com/in/fl%C3%A1vio-henrique-barbosa-38465938)
- **Email:** flaviohenriquehb777@outlook.com

<!-- Progress: 2/210 - 2022-01-16 20:29:00 -->
<!-- Progress: 3/210 - 2022-01-19 14:04:00 -->
<!-- Progress: 4/210 - 2022-01-23 10:24:00 -->
<!-- Progress: 5/210 - 2022-01-24 19:41:00 -->
<!-- Progress: 6/210 - 2022-01-26 21:07:00 -->
<!-- Progress: 7/210 - 2022-01-27 10:11:00 -->
<!-- Progress: 8/210 - 2022-01-27 13:29:00 -->
<!-- Progress: 9/210 - 2022-01-31 20:01:00 -->
<!-- Progress: 10/210 - 2022-01-31 20:15:00 -->
<!-- Progress: 11/210 - 2022-02-01 16:13:00 -->
<!-- Progress: 12/210 - 2022-02-01 18:53:00 -->
<!-- Progress: 13/210 - 2022-02-02 21:24:00 -->
<!-- Progress: 14/210 - 2022-02-03 13:59:00 -->
<!-- Progress: 15/210 - 2022-02-04 08:03:00 -->
<!-- Progress: 16/210 - 2022-02-04 13:56:00 -->
<!-- Progress: 17/210 - 2022-02-04 15:54:00 -->
<!-- Progress: 18/210 - 2022-02-04 16:31:00 -->
<!-- Progress: 19/210 - 2022-02-04 17:20:00 -->
<!-- Progress: 20/210 - 2022-02-06 13:00:00 -->
<!-- Progress: 21/210 - 2022-02-07 09:53:00 -->
<!-- Progress: 22/210 - 2022-02-07 10:59:00 -->
<!-- Progress: 23/210 - 2022-02-10 10:32:00 -->
<!-- Progress: 24/210 - 2022-02-10 20:07:00 -->
<!-- Progress: 25/210 - 2022-02-11 12:10:00 -->
<!-- Progress: 26/210 - 2022-02-11 13:30:00 -->
<!-- Progress: 27/210 - 2022-02-11 16:19:00 -->
<!-- Progress: 28/210 - 2022-02-13 16:29:00 -->
<!-- Progress: 29/210 - 2022-02-14 09:00:00 -->
<!-- Progress: 30/210 - 2022-02-14 11:02:00 -->
<!-- Progress: 31/210 - 2022-02-14 12:47:00 -->
<!-- Progress: 32/210 - 2022-02-14 15:24:00 -->
<!-- Progress: 33/210 - 2022-02-16 17:36:00 -->
<!-- Progress: 34/210 - 2022-02-17 15:01:00 -->
<!-- Progress: 35/210 - 2022-02-17 17:32:00 -->
<!-- Progress: 36/210 - 2022-02-18 13:27:00 -->
<!-- Progress: 37/210 - 2022-02-19 14:07:00 -->
<!-- Progress: 38/210 - 2022-02-19 14:11:00 -->
<!-- Progress: 39/210 - 2022-02-19 17:30:00 -->
<!-- Progress: 40/210 - 2022-02-20 10:16:00 -->
<!-- Progress: 41/210 - 2022-02-22 10:49:00 -->
<!-- Progress: 42/210 - 2022-02-23 09:38:00 -->
<!-- Progress: 43/210 - 2022-02-23 10:34:00 -->
<!-- Progress: 44/210 - 2022-02-23 15:17:00 -->
<!-- Progress: 45/210 - 2022-02-23 15:19:00 -->
<!-- Progress: 46/210 - 2022-02-24 11:23:00 -->
<!-- Progress: 47/210 - 2022-02-24 15:46:00 -->
<!-- Progress: 48/210 - 2022-02-24 18:55:00 -->
<!-- Progress: 49/210 - 2022-02-25 10:43:00 -->
<!-- Progress: 50/210 - 2022-02-25 18:03:00 -->
<!-- Progress: 51/210 - 2022-02-26 18:04:00 -->
<!-- Progress: 52/210 - 2022-02-26 20:03:00 -->
<!-- Progress: 53/210 - 2022-02-27 09:24:00 -->
<!-- Progress: 54/210 - 2022-02-28 11:44:00 -->
<!-- Progress: 55/210 - 2022-03-02 11:46:00 -->
<!-- Progress: 56/210 - 2022-03-02 15:56:00 -->
<!-- Progress: 57/210 - 2022-03-02 18:13:00 -->
<!-- Progress: 58/210 - 2022-03-02 21:04:00 -->
<!-- Progress: 59/210 - 2022-03-03 09:59:00 -->
<!-- Progress: 60/210 - 2022-03-03 12:07:00 -->
<!-- Progress: 61/210 - 2022-03-03 12:24:00 -->
<!-- Progress: 62/210 - 2022-03-03 13:07:00 -->
<!-- Progress: 63/210 - 2022-03-03 13:25:00 -->
<!-- Progress: 64/210 - 2022-03-03 18:55:00 -->
<!-- Progress: 65/210 - 2022-03-04 09:05:00 -->
<!-- Progress: 66/210 - 2022-03-04 11:29:00 -->
<!-- Progress: 67/210 - 2022-03-04 11:31:00 -->
<!-- Progress: 68/210 - 2022-03-04 11:49:00 -->
<!-- Progress: 69/210 - 2022-03-04 20:43:00 -->
<!-- Progress: 70/210 - 2022-03-05 10:26:00 -->
<!-- Progress: 71/210 - 2022-03-06 18:17:00 -->
<!-- Progress: 72/210 - 2022-03-06 19:32:00 -->
<!-- Progress: 73/210 - 2022-03-07 14:37:00 -->
<!-- Progress: 74/210 - 2022-03-08 18:49:00 -->
<!-- Progress: 75/210 - 2022-03-08 20:07:00 -->
<!-- Progress: 76/210 - 2022-03-08 20:30:00 -->
<!-- Progress: 77/210 - 2022-03-08 20:49:00 -->
<!-- Progress: 78/210 - 2022-03-09 08:35:00 -->
<!-- Progress: 79/210 - 2022-03-09 09:39:00 -->
<!-- Progress: 80/210 - 2022-03-09 10:27:00 -->
<!-- Progress: 81/210 - 2022-03-09 13:59:00 -->
<!-- Progress: 82/210 - 2022-03-10 09:30:00 -->
<!-- Progress: 83/210 - 2022-03-10 13:14:00 -->
<!-- Progress: 84/210 - 2022-03-11 13:14:00 -->
<!-- Progress: 85/210 - 2022-03-11 13:42:00 -->
<!-- Progress: 86/210 - 2022-03-12 10:15:00 -->
<!-- Progress: 87/210 - 2022-03-12 16:03:00 -->
<!-- Progress: 88/210 - 2022-03-12 20:45:00 -->
<!-- Progress: 89/210 - 2022-03-13 09:51:00 -->
<!-- Progress: 90/210 - 2022-03-13 12:24:00 -->
<!-- Progress: 91/210 - 2022-03-13 16:02:00 -->
<!-- Progress: 92/210 - 2022-03-14 09:40:00 -->
<!-- Progress: 93/210 - 2022-03-14 11:16:00 -->
<!-- Progress: 94/210 - 2022-03-14 15:09:00 -->
<!-- Progress: 95/210 - 2022-03-14 16:56:00 -->
<!-- Progress: 96/210 - 2022-03-15 21:10:00 -->
<!-- Progress: 97/210 - 2022-03-16 12:10:00 -->
<!-- Progress: 98/210 - 2022-03-18 08:28:00 -->
<!-- Progress: 99/210 - 2022-03-18 09:32:00 -->
<!-- Progress: 100/210 - 2022-03-18 13:11:00 -->
<!-- Progress: 101/210 - 2022-03-20 08:37:00 -->
<!-- Progress: 102/210 - 2022-03-20 10:19:00 -->
<!-- Progress: 103/210 - 2022-03-20 18:47:00 -->
<!-- Progress: 104/210 - 2022-03-20 20:00:00 -->
<!-- Progress: 105/210 - 2022-03-21 18:57:00 -->
<!-- Progress: 106/210 - 2022-03-22 13:44:00 -->
<!-- Progress: 107/210 - 2022-03-22 16:10:00 -->
<!-- Progress: 108/210 - 2022-03-22 16:15:00 -->
<!-- Progress: 109/210 - 2022-03-23 12:44:00 -->
<!-- Progress: 110/210 - 2022-03-24 10:03:00 -->
<!-- Progress: 111/210 - 2022-03-24 10:19:00 -->
<!-- Progress: 112/210 - 2022-03-24 12:12:00 -->
<!-- Progress: 113/210 - 2022-03-24 13:49:00 -->
<!-- Progress: 114/210 - 2022-03-24 14:00:00 -->
<!-- Progress: 115/210 - 2022-03-24 15:03:00 -->
<!-- Progress: 116/210 - 2022-03-24 17:47:00 -->
<!-- Progress: 117/210 - 2022-03-25 10:00:00 -->
<!-- Progress: 118/210 - 2022-03-26 09:16:00 -->
<!-- Progress: 119/210 - 2022-03-26 13:07:00 -->
<!-- Progress: 120/210 - 2022-03-26 14:40:00 -->
<!-- Progress: 121/210 - 2022-03-26 17:57:00 -->
<!-- Progress: 122/210 - 2022-03-27 12:52:00 -->
<!-- Progress: 123/210 - 2022-03-27 15:03:00 -->
<!-- Progress: 124/210 - 2022-03-27 17:58:00 -->
<!-- Progress: 125/210 - 2022-03-28 08:33:00 -->
<!-- Progress: 126/210 - 2022-03-28 12:21:00 -->
<!-- Progress: 127/210 - 2022-03-28 14:31:00 -->
<!-- Progress: 128/210 - 2022-03-28 15:54:00 -->
<!-- Progress: 129/210 - 2022-03-28 18:03:00 -->
<!-- Progress: 130/210 - 2022-03-28 19:23:00 -->
<!-- Progress: 131/210 - 2022-03-29 16:41:00 -->
<!-- Progress: 132/210 - 2022-03-29 18:07:00 -->
<!-- Progress: 133/210 - 2022-03-30 19:37:00 -->
<!-- Progress: 134/210 - 2022-03-30 21:31:00 -->
<!-- Progress: 135/210 - 2022-03-31 15:51:00 -->
<!-- Progress: 136/210 - 2022-04-01 13:45:00 -->
<!-- Progress: 137/210 - 2022-04-01 17:25:00 -->
<!-- Progress: 138/210 - 2022-04-02 10:30:00 -->
<!-- Progress: 139/210 - 2022-04-02 15:14:00 -->
<!-- Progress: 140/210 - 2022-04-02 20:49:00 -->
<!-- Progress: 141/210 - 2022-04-04 15:36:00 -->
<!-- Progress: 142/210 - 2022-04-05 12:39:00 -->
<!-- Progress: 143/210 - 2022-04-05 16:08:00 -->
<!-- Progress: 144/210 - 2022-04-05 16:49:00 -->
<!-- Progress: 145/210 - 2022-04-06 19:32:00 -->
<!-- Progress: 146/210 - 2022-04-07 12:28:00 -->
<!-- Progress: 147/210 - 2022-04-09 13:31:00 -->
<!-- Progress: 148/210 - 2022-04-09 14:14:00 -->
<!-- Progress: 149/210 - 2022-04-09 16:31:00 -->
<!-- Progress: 150/210 - 2022-04-10 17:26:00 -->
<!-- Progress: 151/210 - 2022-04-10 18:59:00 -->
<!-- Progress: 152/210 - 2022-04-11 09:17:00 -->
<!-- Progress: 153/210 - 2022-04-11 12:28:00 -->
<!-- Progress: 154/210 - 2022-04-13 11:33:00 -->
<!-- Progress: 155/210 - 2022-04-14 08:26:00 -->
<!-- Progress: 156/210 - 2022-04-14 12:36:00 -->
<!-- Progress: 157/210 - 2022-04-14 18:34:00 -->
<!-- Progress: 158/210 - 2022-04-15 12:08:00 -->
<!-- Progress: 159/210 - 2022-04-15 12:25:00 -->
<!-- Progress: 160/210 - 2022-04-15 18:20:00 -->
<!-- Progress: 161/210 - 2022-04-15 20:39:00 -->
<!-- Progress: 162/210 - 2022-04-16 13:52:00 -->
<!-- Progress: 163/210 - 2022-04-16 18:22:00 -->
<!-- Progress: 164/210 - 2022-04-17 16:04:00 -->
<!-- Progress: 165/210 - 2022-04-19 12:02:00 -->
<!-- Progress: 166/210 - 2022-04-20 09:31:00 -->
<!-- Progress: 167/210 - 2022-04-20 21:44:00 -->
<!-- Progress: 168/210 - 2022-04-21 15:40:00 -->
<!-- Progress: 169/210 - 2022-04-22 09:37:00 -->
<!-- Progress: 170/210 - 2022-04-22 13:41:00 -->
<!-- Progress: 171/210 - 2022-04-24 08:14:00 -->
<!-- Progress: 172/210 - 2022-04-24 12:12:00 -->
<!-- Progress: 173/210 - 2022-04-24 19:50:00 -->
<!-- Progress: 174/210 - 2022-04-25 13:07:00 -->
<!-- Progress: 175/210 - 2022-04-26 16:59:00 -->
<!-- Progress: 176/210 - 2022-04-26 21:49:00 -->
<!-- Progress: 177/210 - 2022-04-27 13:42:00 -->