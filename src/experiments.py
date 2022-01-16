import json
import os
import warnings
import re
from urllib.parse import urlparse
from pathlib import Path
from typing import Any, Callable, Optional
from projeto_maximizacao_receita.config.paths import (
    DADOS_AMOR_A_CAKES,
    MODELS_DIR,
    PROJECT_ROOT,
)

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler

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
    if load_dotenv is not None:
        load_dotenv()

    repo_name = os.getenv("DAGSHUB_REPO")
    owner = os.getenv("DAGSHUB_OWNER")
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    enable_remote = os.getenv("ENABLE_REMOTE_MLFLOW")

    def _truthy(val: Optional[str]) -> bool:
        return str(val).lower() in {"1", "true", "yes", "on"} if val else False

    allow_remote = _truthy(enable_remote)

    # Sanitiza variáveis de ambiente de MLflow que possam apontar para drives Windows
    for var in [
        "MLFLOW_ARTIFACT_URI",
        "MLFLOW_ARTIFACT_LOCATION",
        "MLFLOW_ARTIFACTS_DIR",
    ]:
        val = os.getenv(var)
        if not val:
            continue
        parsed_val = urlparse(val)
        if parsed_val.scheme == "file":
            if os.name != "nt" and re.match(r"^/[A-Za-z]:", parsed_val.path or ""):
                os.environ.pop(var, None)
        else:
            if os.name != "nt" and (
                re.match(r"^[A-Za-z]:(\\|/)", val) or re.match(r"^/[A-Za-z]:", val)
            ):
                os.environ.pop(var, None)

    if mlflow_uri:
        parsed = urlparse(mlflow_uri)
        if parsed.scheme in {"http", "https"}:
            if allow_remote:
                mlflow.set_tracking_uri(mlflow_uri)
                return "custom"
            else:
                warnings.warn(
                    "MLFLOW_TRACKING_URI remoto ignorado no CI; usando MLflow local."
                )
        elif parsed.scheme == "file":
            if os.name != "nt" and re.match(r"^/[A-Za-z]:", parsed.path or ""):
                warnings.warn(
                    "MLFLOW_TRACKING_URI aponta para drive Windows; ignorando e usando MLflow local."
                )
            else:
                mlflow.set_tracking_uri(mlflow_uri)
                return "custom"
        else:
            if os.name != "nt" and (
                re.match(r"^[A-Za-z]:(\\|/)", mlflow_uri)
                or re.match(r"^/[A-Za-z]:", mlflow_uri)
            ):
                warnings.warn(
                    "MLFLOW_TRACKING_URI parece caminho Windows; ignorando e usando MLflow local."
                )
            else:
                try:
                    mlflow.set_tracking_uri(mlflow_uri)
                    return "custom"
                except Exception:
                    warnings.warn("MLFLOW_TRACKING_URI inválido; usando MLflow local.")

    if allow_remote and dagshub is not None and repo_name and owner and dagshub_token:
        try:
            dagshub.init(repo_name, owner, mlflow=True)
            return "dagshub"
        except Exception:
            warnings.warn("Falha ao inicializar DagsHub; usando MLflow local.")

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


def evaluate_model(model_name: str, model, X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model", model_name)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))

        mlflow.log_metrics({"rmse": rmse, "r2": r2})
        return {"name": model_name, "model": model, "rmse": rmse, "r2": r2}


def main():
    setup_tracking()
    df = load_and_prepare_data(DADOS_AMOR_A_CAKES)

    # Carregar parâmetros
    params_path = PROJECT_ROOT / "params.yaml"
    if params_path.exists():
        with open(params_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        exp = cfg.get("experiments", {})
        ridge_alphas = exp.get("ridge_alphas", [0.1, 1.0, 10.0])
        lasso_alphas = exp.get("lasso_alphas", [0.1, 1.0, 10.0])
        rf_n_estimators = exp.get("rf_n_estimators", [100, 300])
        gb_learning_rates = exp.get("gb_learning_rates", [0.05, 0.1])
        test_size = exp.get("test_size", 0.2)
        random_state = exp.get("random_state", RANDOM_STATE)
    else:
        ridge_alphas = [0.1, 1.0, 10.0]
        lasso_alphas = [0.1, 1.0, 10.0]
        rf_n_estimators = [100, 300]
        gb_learning_rates = [0.05, 0.1]
        test_size = 0.2
        random_state = RANDOM_STATE

    features = ["PrecoVenda_scaled", "PrecoOriginal_scaled", "Desconto_scaled"]
    target = "VendaQtd_scaled"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    candidates = []

    # Baseline
    candidates.append(("LinearRegression", LinearRegression()))

    # Regularização
    for alpha in ridge_alphas:
        candidates.append(
            (f"Ridge_alpha_{alpha}", Ridge(alpha=alpha, random_state=RANDOM_STATE))
        )
    for alpha in lasso_alphas:
        candidates.append(
            (f"Lasso_alpha_{alpha}", Lasso(alpha=alpha, random_state=RANDOM_STATE))
        )

    # Ensemble simples
    for n_estimators in rf_n_estimators:
        candidates.append(
            (
                f"RandomForest_{n_estimators}",
                RandomForestRegressor(
                    n_estimators=n_estimators, random_state=RANDOM_STATE
                ),
            )
        )
    for learning_rate in gb_learning_rates:
        candidates.append(
            (
                f"GradientBoosting_lr_{learning_rate}",
                GradientBoostingRegressor(
                    learning_rate=learning_rate, random_state=RANDOM_STATE
                ),
            )
        )

    results = []
    for name, model in candidates:
        res = evaluate_model(name, model, X_train, y_train, X_test, y_test)
        results.append(res)

    # Escolher melhor por RMSE (menor)
    best = min(results, key=lambda r: r["rmse"])

    MODELS_DIR.mkdir(exist_ok=True)
    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    best_model_path = MODELS_DIR / "model_best.joblib"
    best_metrics_path = artifacts_dir / "metrics_best.json"

    joblib.dump(best["model"], best_model_path)
    with open(best_metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {"model": best["name"], "rmse": best["rmse"], "r2": best["r2"]},
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Registrar modelo no MLflow para ficar explícito no DagsHub (Models)
    try:
        import mlflow.sklearn as mls

        with mlflow.start_run(run_name=f"best_model_{best['name']}"):
            mlflow.log_params(
                {
                    "best_model": best["name"],
                    "rmse": best["rmse"],
                    "r2": best["r2"],
                }
            )
            mls.log_model(
                best["model"], artifact_path="model", registered_model_name="best-model"
            )
    except Exception as e:
        warnings.warn(f"Falha ao registrar modelo no MLflow: {e}")

    print(
        f"Melhor modelo: {best['name']} | RMSE={best['rmse']:.4f} R2={best['r2']:.4f}"
    )
    print(f"Salvo em: {best_model_path}")


if __name__ == "__main__":
    main()
