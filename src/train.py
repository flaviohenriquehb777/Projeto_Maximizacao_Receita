import json
import os
import warnings
import re
from urllib.parse import urlparse
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from typing import Any, Callable, Optional
from projeto_maximizacao_receita.config.paths import (
    DADOS_AMOR_A_CAKES,
    MODELS_DIR,
    PROJECT_ROOT,
)

dagshub: Optional[Any]
try:
    import dagshub as _dagshub

    dagshub = _dagshub
except Exception:
    dagshub = None

load_dotenv: Optional[Callable[..., bool]]
try:
    from dotenv import load_dotenv as _load_dotenv

    load_dotenv = _load_dotenv
except Exception:
    load_dotenv = None


RANDOM_STATE = 42


def setup_tracking():
    """Configure MLflow tracking (local by default, DagsHub if env is present)."""
    if load_dotenv is not None:
        load_dotenv()

    repo_name = os.getenv("DAGSHUB_REPO")
    owner = os.getenv("DAGSHUB_OWNER")
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")

    if mlflow_uri:
        parsed = urlparse(mlflow_uri)
        # Accept explicit URIs (http/https/file). Otherwise guard against Windows paths on Linux runners.
        if parsed.scheme in {"http", "https", "file"}:
            mlflow.set_tracking_uri(mlflow_uri)
            return "custom"
        # Ignore Windows-style paths when not on Windows (e.g., C:\ or C:/)
        if os.name != "nt" and re.match(r"^[A-Za-z]:(\\|/)", mlflow_uri):
            warnings.warn(
                "MLFLOW_TRACKING_URI parece caminho Windows; ignorando no CI e usando MLflow local."
            )
        else:
            try:
                mlflow.set_tracking_uri(mlflow_uri)
                return "custom"
            except Exception:
                warnings.warn("MLFLOW_TRACKING_URI inválido; usando MLflow local.")

    if dagshub is not None and repo_name and owner:
        try:
            dagshub.init(repo_name, owner, mlflow=True)
            return "dagshub"
        except Exception:
            warnings.warn("Falha ao inicializar DagsHub; usando MLflow local.")

    # Default: local mlflow in ./mlruns
    mlflow.set_tracking_uri((PROJECT_ROOT / "mlruns").as_uri())
    return "local"


def _synthetic_df(n_rows: int = 200, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    preco_venda = rng.normal(50, 10, size=n_rows)
    preco_original = preco_venda + rng.normal(5, 3, size=n_rows)
    desconto = rng.normal(0.1, 0.05, size=n_rows)
    venda_qtd = 0.5 * preco_venda - 0.3 * desconto + rng.normal(0, 1, size=n_rows)

    df = pd.DataFrame(
        {
            "PrecoVenda": preco_venda,
            "PrecoOriginal": preco_original,
            "Desconto": desconto,
            "VendaQtd": venda_qtd,
        }
    )
    return df


def load_and_prepare_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        warnings.warn(
            f"Dataset não encontrado em '{path}'. Usando dados sintéticos para CI."
        )
        df = _synthetic_df()
    else:
        df = pd.read_excel(path)

    # Escalonamento conforme notebooks
    minmax_cols = ["PrecoVenda", "PrecoOriginal", "VendaQtd"]
    robust_cols = ["Desconto"]

    # Imputação simples para valores ausentes
    df[minmax_cols + robust_cols] = df[minmax_cols + robust_cols].apply(
        lambda s: s.fillna(s.median())
    )

    mm = MinMaxScaler()
    rb = RobustScaler()

    df[minmax_cols] = mm.fit_transform(df[minmax_cols])
    df[robust_cols] = rb.fit_transform(df[robust_cols])

    df = df.rename(
        columns={
            "PrecoVenda": "PrecoVenda_scaled",
            "PrecoOriginal": "PrecoOriginal_scaled",
            "Desconto": "Desconto_scaled",
            "VendaQtd": "VendaQtd_scaled",
        }
    )

    return df


def train_and_evaluate(df: pd.DataFrame):
    features = ["PrecoVenda_scaled", "PrecoOriginal_scaled", "Desconto_scaled"]
    target = "VendaQtd_scaled"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipeline = Pipeline(steps=[("LinearRegression", LinearRegression())])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    return pipeline, {"rmse": rmse, "r2": r2}


def save_artifacts(model, metrics: dict):
    MODELS_DIR.mkdir(exist_ok=True)
    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    model_path = MODELS_DIR / "model_linear.joblib"
    metrics_path = artifacts_dir / "metrics.json"

    joblib.dump(model, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return model_path, metrics_path


def main():
    tracking_mode = setup_tracking()

    with mlflow.start_run(run_name=f"train_linear_regression_{tracking_mode}"):
        mlflow.log_params(
            {
                "model": "LinearRegression",
                "random_state": RANDOM_STATE,
                "scaler_minmax": ["PrecoVenda", "PrecoOriginal", "VendaQtd"],
                "scaler_robust": ["Desconto"],
            }
        )

        df = load_and_prepare_data(DADOS_AMOR_A_CAKES)
        mlflow.log_param("dataset_rows", int(df.shape[0]))

        model, metrics = train_and_evaluate(df)

        mlflow.log_metrics(metrics)

        model_path, metrics_path = save_artifacts(model, metrics)
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(metrics_path))

        print("Treino concluído.")
        print(f"RMSE: {metrics['rmse']:.4f} | R2: {metrics['r2']:.4f}")
        print(f"Modelo salvo em: {model_path}")


if __name__ == "__main__":
    main()
