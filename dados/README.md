Este diretório armazena dados locais não versionados.

Arquivos como `dataset_cafeterias_rj.xlsx` são ignorados pelo Git (ver `.gitignore`).

Como obter o dataset:
- Coloque o arquivo `dataset_cafeterias_rj.xlsx` em `dados/` antes de executar o pipeline.
- Caso não possua o dataset, o pipeline pode gerar dados sintéticos para testes.

Boas práticas:
- Não versionar dados sensíveis em GitHub.
- Usar storage remoto (ex.: DagsHub datasets, S3) ou DVC para controle de versões de dados.