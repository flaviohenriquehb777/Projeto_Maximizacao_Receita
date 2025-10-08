import os
import json
import warnings
from pathlib import Path
import sys

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

import mlflow

try:
    import dagshub
except Exception:
    dagshub = None

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config.paths import DADOS_AMOR_A_CAKES, MODELS_DIR, PROJECT_ROOT


RANDOM_STATE = 42


def setup_tracking():
    if load_dotenv is not None:
        load_dotenv()

    repo_name = os.getenv("DAGSHUB_REPO")
    owner = os.getenv("DAGSHUB_OWNER")
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")

    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
        return "custom"

    if dagshub and repo_name and owner:
        try:
            dagshub.init(repo_name, owner, mlflow=True)
            return "dagshub"
        except Exception:
            warnings.warn("Falha ao inicializar DagsHub; usando MLflow local.")

    mlflow.set_tracking_uri((PROJECT_ROOT / "mlruns").as_uri())
    return "local"


def load_and_prepare_data(path: Path) -> pd.DataFrame:
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
    tracking_mode = setup_tracking()
    df = load_and_prepare_data(DADOS_AMOR_A_CAKES)

    features = ["PrecoVenda_scaled", "PrecoOriginal_scaled", "Desconto_scaled"]
    target = "VendaQtd_scaled"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    candidates = []

    # Baseline
    candidates.append(("LinearRegression", LinearRegression()))

    # Regularização
    for alpha in [0.1, 1.0, 10.0]:
        candidates.append((f"Ridge_alpha_{alpha}", Ridge(alpha=alpha, random_state=RANDOM_STATE)))
        candidates.append((f"Lasso_alpha_{alpha}", Lasso(alpha=alpha, random_state=RANDOM_STATE)))

    # Ensemble simples
    for n_estimators in [100, 300]:
        candidates.append((f"RandomForest_{n_estimators}", RandomForestRegressor(n_estimators=n_estimators, random_state=RANDOM_STATE)))
    for learning_rate in [0.05, 0.1]:
        candidates.append((f"GradientBoosting_lr_{learning_rate}", GradientBoostingRegressor(learning_rate=learning_rate, random_state=RANDOM_STATE)))

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
        json.dump({"model": best["name"], "rmse": best["rmse"], "r2": best["r2"]}, f, ensure_ascii=False, indent=2)

    print(f"Melhor modelo: {best['name']} | RMSE={best['rmse']:.4f} R2={best['r2']:.4f}")
    print(f"Salvo em: {best_model_path}")


if __name__ == "__main__":
    main()