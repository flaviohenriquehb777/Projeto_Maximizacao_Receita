import json
import os
from pathlib import Path

from src.config.paths import MODELS_DIR
from src.modeling.train_pipeline import main


def _approx_equal(actual: float, expected: float, tol_abs: float, tol_rel: float) -> bool:
    if expected == 0:
        return abs(actual) <= tol_abs
    return abs(actual - expected) <= max(tol_abs, tol_rel * abs(expected))


def test_training_produces_artifacts_and_metrics_snapshot():
    # Executa o pipeline uma vez
    main()

    # Verifica JSON para UI (sem depender de .pkl nos testes)
    json_path = MODELS_DIR / 'model_linear.json'
    assert json_path.exists(), "Model JSON para UI não foi criado."

    payload = json.loads(json_path.read_text(encoding='utf-8'))
    assert 'coefficients' in payload and isinstance(payload['coefficients'], dict), "JSON sem coefficients."
    assert 'intercept' in payload, "JSON sem intercept."

    # Verifica snapshot de métricas
    snapshot_path = MODELS_DIR / 'metrics_snapshot.json'
    assert snapshot_path.exists(), "metrics_snapshot.json não foi criado."
    snapshot = json.loads(snapshot_path.read_text(encoding='utf-8'))
    assert 'metrics' in snapshot and isinstance(snapshot['metrics'], dict), "Snapshot sem métricas."

    is_ci = os.getenv('CI', '').lower() == 'true'
    if is_ci:
        # Em CI, o dataset original pode não estar versionado; validamos estrutura e sanidade.
        required = ['cv_mae','cv_rmse','cv_r2','holdout_mae','holdout_rmse','holdout_r2']
        for k in required:
            assert k in snapshot['metrics'], f"Métrica {k} ausente no snapshot."
            v = float(snapshot['metrics'][k])
            assert not (v is None or (v != v)), f"Métrica {k} inválida (NaN)."
        # Limites básicos
        assert snapshot['metrics']['cv_mae'] >= 0 and snapshot['metrics']['cv_rmse'] >= 0
        assert -1.0 <= snapshot['metrics']['cv_r2'] <= 1.0
        assert snapshot['metrics']['holdout_mae'] >= 0 and snapshot['metrics']['holdout_rmse'] >= 0
        assert -1.0 <= snapshot['metrics']['holdout_r2'] <= 1.0
    else:
        # Localmente com dados originais, comparamos com snapshot esperado com tolerância.
        expected_path = Path('tests') / 'expected_metrics.json'
        assert expected_path.exists(), "tests/expected_metrics.json ausente."
        expected = json.loads(expected_path.read_text(encoding='utf-8'))

        tol_abs = float(expected.get('tolerance', {}).get('abs', 1e-6))
        tol_rel = float(expected.get('tolerance', {}).get('rel', 0.05))
        for key, exp_val in expected.get('metrics', {}).items():
            assert key in snapshot['metrics'], f"Métrica {key} não encontrada no snapshot."
            act_val = float(snapshot['metrics'][key])
            assert _approx_equal(act_val, float(exp_val), tol_abs, tol_rel), (
                f"Métrica {key} fora da tolerância. Esperado={exp_val}, Atual={act_val}, "
                f"tol_abs={tol_abs}, tol_rel={tol_rel}"
            )