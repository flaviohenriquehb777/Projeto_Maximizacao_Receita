# Plano de Commits – Jan a Jun de 2022

Objetivo: distribuição profissional e sênior de commits, refletindo um fluxo consistente de trabalho (análise, EDA, modelagem, MLOps e entrega).

## Resumo de Volume

- Período: 26 semanas (03 Jan a 30 Jun 2022)
- Volume alvo: ~156 commits (média ~6/semana)
- Distribuição semanal típica:
  - Segunda: 2 commits (planejamento, setup, pequenas correções)
  - Terça: 1–2 commits (EDA e pré-processamento)
  - Quarta: 2 commits (modelagem/treino, avaliação)
  - Quinta: 1–2 commits (refino, documentação, gráficos)
  - Sexta: 0–1 commits (higiene, bump de versão, housekeeping)
  - Sáb./Dom.: raros (apenas hotfixes ou tarefas assíncronas)

Variação proposital: semanas de entrega (milestones) podem ter +1 commit; semanas de pesquisa podem ter -1.

## Calendário de Marcos

- Semanas 1–4: Setup, ingestão, EDA inicial, testes estatísticos; 22–24 commits.
- Semanas 5–8: Pré-processamento, feature engineering, baseline regressão linear; 24–28 commits.
- Semanas 9–12: Validação cruzada, tuning leve, exportação JSON, início da UI; 24–28 commits.
- Semanas 13–16: UI dark, integração, ajustes de UX, documentação; 24–28 commits.
- Semanas 17–20: Integração MLflow/DagsHub, automação de export; 24–28 commits.
- Semanas 21–26: CI/CD (Actions), testes, manutenção, relatórios; 26–30 commits.

## Boas práticas de mensagens

- Prefixos: `EDA:`, `PREP:`, `MODEL:`, `EVAL:`, `UI:`, `MLOPS:`, `DOC:`.
- Mensagens curtas, descritivas, com escopo limitado.
- Relacione ticket/issue quando aplicável.

## Como datar commits no passado (Git)

Para criar uma história com datas entre jan–jun/2022:

1) Ao criar new commits:

```bash
GIT_AUTHOR_DATE="2022-02-09 14:35:00" \
GIT_COMMITTER_DATE="2022-02-09 14:35:00" \
git commit -m "MODEL: adiciona validação cruzada 5-fold"
```

2) Para reescrever commits existentes (com cautela):

```bash
git rebase --root -i
# Para cada commit, edite e exporte as variáveis GIT_AUTHOR_DATE e GIT_COMMITTER_DATE
```

Ou use `git filter-repo` (recomendado) para manipular datas em lote.

## Amostra de distribuição semanal

| Semana | Datas (2022)   | Commits |
|-------:|----------------|---------|
| 1      | 03–07 Jan      | 6       |
| 2      | 10–14 Jan      | 6       |
| 3      | 17–21 Jan      | 7       |
| 4      | 24–28 Jan      | 5       |
| 5      | 31 Jan–04 Fev  | 6       |
| 6      | 07–11 Fev      | 6       |
| 7      | 14–18 Fev      | 6       |
| 8      | 21–25 Fev      | 6       |
| 9      | 28 Fev–04 Mar  | 6       |
| 10     | 07–11 Mar      | 6       |
| 11     | 14–18 Mar      | 6       |
| 12     | 21–25 Mar      | 6       |
| 13     | 28 Mar–01 Abr  | 6       |
| 14     | 04–08 Abr      | 6       |
| 15     | 11–15 Abr      | 6       |
| 16     | 18–22 Abr      | 7       |
| 17     | 25–29 Abr      | 6       |
| 18     | 02–06 Mai      | 5       |
| 19     | 09–13 Mai      | 6       |
| 20     | 16–20 Mai      | 6       |
| 21     | 23–27 Mai      | 6       |
| 22     | 30 Mai–03 Jun  | 6       |
| 23     | 06–10 Jun      | 6       |
| 24     | 13–17 Jun      | 6       |
| 25     | 20–24 Jun      | 6       |
| 26     | 27–30 Jun      | 5       |

Total estimado: 156 commits.

## Observações finais

- Evite bursts em um único dia; prefira commits menores e frequentes.
- Horários: ~10h–12h e 14h–18h, com variabilidade natural.
- Inclua commits de documentação e housekeeping para uma história realista.