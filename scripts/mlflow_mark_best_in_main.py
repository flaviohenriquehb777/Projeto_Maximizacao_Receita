import os
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient
import dagshub


def mark_best_in_experiment(
    experiment_name: str,
    best_label: str,
    tracking_uri: Optional[str] = None,
):
    repo = os.environ.get('DAGSHUB_REPO')
    owner = os.environ.get('DAGSHUB_OWNER')
    if not repo or not owner:
        raise RuntimeError('Ambiente DagsHub não configurado (DAGSHUB_REPO/DAGSHUB_OWNER)')

    dagshub.init(repo, owner, mlflow=True)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if not exp:
        raise RuntimeError(f'Experimento não encontrado: {experiment_name}')

    # Buscar runs
    runs = client.search_runs(exp.experiment_id, order_by=["attributes.start_time DESC"], max_results=1000)

    # 1) Tentar localizar por label explícito
    target_run = None
    for r in runs:
        tags = r.data.tags or {}
        params = r.data.params or {}
        run_name = tags.get('mlflow.runName') or getattr(r.info, 'run_name', None)
        if run_name == best_label or tags.get('best_model') == best_label or params.get('model_name') == best_label:
            target_run = r
            break

    # 2) Fallback: escolher melhor por métricas (prioriza lucro esperado; senão menor RMSE)
    if not target_run:
        def score_run(run):
            m = run.data.metrics or {}
            # prioriza métricas de lucro
            profit_keys = [k for k in m.keys() if 'profit' in k.lower()]
            if profit_keys:
                return max(m.get(k, float('-inf')) for k in profit_keys)
            # senão, usa RMSE (menor é melhor)
            rmse = None
            for k in ['cv_rmse', 'holdout_rmse', 'rmse']:
                if k in m:
                    rmse = m[k]
                    break
            return float('inf') if rmse is None else -rmse  # maximizar (-RMSE)

        scored = [(score_run(r), r) for r in runs]
        scored = [t for t in scored if t[0] != float('-inf') and t[0] != float('inf')]
        if scored:
            scored.sort(key=lambda t: t[0], reverse=True)
            target_run = scored[0][1]

    if not target_run:
        raise RuntimeError(f'Não foi possível identificar o run vencedor no experimento "{experiment_name}".')

    # Detectar label a partir do run
    tags = target_run.data.tags or {}
    params = target_run.data.params or {}
    detected_label = tags.get('model_name') or params.get('model_name') or best_label

    # Marcar o run encontrado
    client.set_tag(target_run.info.run_id, 'is_best', 'true')
    client.set_tag(target_run.info.run_id, 'best_model', detected_label)
    client.set_tag(target_run.info.run_id, 'mlflow.runName', 'best_model')
    print('[MLflow] Best run marcado:', target_run.info.run_id, '| label:', detected_label)


def main():
    best_label = os.environ.get('BEST_MODEL_LABEL') or 'GradientBoostingRegressor'
    experiment_name = os.environ.get('MLFLOW_EXPERIMENT_NAME') or 'Projeto_Maximizacao_Receita_2022_Jan_Jun'
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')
    mark_best_in_experiment(experiment_name, best_label, tracking_uri)


if __name__ == '__main__':
    main()