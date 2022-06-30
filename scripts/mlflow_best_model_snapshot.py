import os
import json
from pathlib import Path

import mlflow
import dagshub


def main():
    # 1) Confirmar dataset local
    dataset_path = Path('dados/dataset_cafeterias_rj.xlsx')
    print('[CHECK] Dataset existe?', dataset_path.exists(), '->', dataset_path)

    # 2) Inicializar tracking remoto e criar/selecionar experimento 'best_model'
    repo = os.environ.get('DAGSHUB_REPO')
    owner = os.environ.get('DAGSHUB_OWNER')
    if not repo or not owner:
        raise RuntimeError('Ambiente DagsHub não configurado (DAGSHUB_REPO/DAGSHUB_OWNER)')

    dagshub.init(repo, owner, mlflow=True)
    tracking_uri_env = os.environ.get('MLFLOW_TRACKING_URI')
    if tracking_uri_env:
        mlflow.set_tracking_uri(tracking_uri_env)

    mlflow.set_experiment('best_model')

    # 3) Carregar snapshot atual dos melhores resultados
    snap_path = Path('models/metrics_snapshot.json')
    if not snap_path.exists():
        raise FileNotFoundError(f'Arquivo não encontrado: {snap_path}')

    with snap_path.open('r', encoding='utf-8') as f:
        snap = json.load(f)

    best_name = snap.get('best_model_name') or snap.get('best_model') or 'unknown'
    metrics = snap.get('metrics') or {}
    features = snap.get('feature_order') or []

    params = {
        'best_model': best_name,
        'feature_count': len(features),
        'period_start': '2022-01-01',
        'period_end': '2022-06-30',
    }

    # 4) Registrar run com tags e artefatos
    # Use o nome do run como 'best_model' para facilitar a identificação
    with mlflow.start_run(run_name="best_model") as run:
        # Parâmetros
        mlflow.log_params(params)

        # Métricas (filtrar valores numéricos simples)
        if isinstance(metrics, dict):
            flat_metrics = {}
            for k, v in metrics.items():
                try:
                    if isinstance(v, (int, float)):
                        flat_metrics[k] = float(v)
                except Exception:
                    pass
            if flat_metrics:
                mlflow.log_metrics(flat_metrics)

        # Tags para destacar o melhor e fixar o nome visível
        mlflow.set_tag('is_best', 'true')
        mlflow.set_tag('best_model', best_name)
        # Algumas UIs usam essa tag especial para exibir o nome do run
        mlflow.set_tag('mlflow.runName', 'best_model')

        # Anexar o snapshot como artefato
        mlflow.log_artifact(str(snap_path), artifact_path='snapshot')

        print('[MLflow] Run ID:', run.info.run_id)

    print('[MLflow] Tracking URI:', mlflow.get_tracking_uri())
    print('[MLflow] Experimento criado/selecionado: best_model')


if __name__ == '__main__':
    main()