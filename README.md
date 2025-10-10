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
  - RMSE: 7.939962493671038e-05
  - R2: 0.9999999030253405
- Melhor Modelo
  - Modelo: LinearRegression
  - RMSE: 7.939962493671038e-05
  - R2: 0.9999999030253405

Fonte: [GitHub Actions](https://github.com/flaviohenriquehb777/Projeto_Maximizacao_Receita/actions/runs/18410895809)

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
