import os
import json
from pathlib import Path
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# SHAP opcional (evita dependências nativas em runners mínimos)
try:
    import shap  # type: ignore
    HAS_SHAP = True
except Exception:
    shap = None  # type: ignore
    HAS_SHAP = False

import mlflow

try:
    import dagshub
except Exception:
    dagshub = None

from src.config.paths import DADOS_AMOR_A_CAKES, MODELS_DIR


@dataclass
class ModelSpec:
    name: str
    estimator: object
    params: dict


def init_tracking():
    """Inicializa MLflow e opcionalmente DagsHub, lendo variáveis de ambiente.

    Suporta ambos padrões de env:
    - DAGSHUB_REPO_OWNER / DAGSHUB_REPO_NAME
    - DAGSHUB_OWNER / DAGSHUB_REPO
    E MLFLOW_TRACKING_URI explícito quando fornecido.
    """
    # Primeiro, configurar tracking URI (com credenciais se disponíveis), depois set_experiment
    owner = os.getenv("DAGSHUB_REPO_OWNER") or os.getenv("DAGSHUB_OWNER")
    repo = os.getenv("DAGSHUB_REPO_NAME") or os.getenv("DAGSHUB_REPO")
    tracking_uri_env = os.getenv("MLFLOW_TRACKING_URI")
    username = os.getenv("MLFLOW_TRACKING_USERNAME") or os.getenv("DAGSHUB_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD") or os.getenv("DAGSHUB_TOKEN")
    token = os.getenv("DAGSHUB_TOKEN")

    # Registrar token no cliente DagsHub antes de inicializar MLflow
    if dagshub is not None and token:
        try:
            import dagshub.auth  # type: ignore
            dagshub.auth.add_token(token)  # type: ignore
        except Exception as e:
            warnings.warn(f"Falha ao registrar token DagsHub: {e}")

    # Inicializar DagsHub (isso pode configurar o MLflow tracking automaticamente)
    if dagshub is not None and owner and repo:
        try:
            dagshub.init(
                repo_owner=owner,
                repo_name=repo,
                mlflow=True,
            )
        except Exception as e:
            warnings.warn(f"Falha ao iniciar DagsHub: {e}")

    # Garantir que MLflow leia credenciais via env
    if username and password:
        os.environ.setdefault("MLFLOW_TRACKING_USERNAME", username)
        os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", password)

    # Se credenciais estiverem presentes, preferir URI com basic auth embutida
    if owner and repo:
        try:
            # Preferir URI sem credenciais; MLflow usará USERNAME/PASSWORD do env
            tracking_uri = f"https://dagshub.com/{owner}/{repo}.mlflow"
            mlflow.set_tracking_uri(tracking_uri)
        except Exception as e:
            warnings.warn(f"Falha ao definir tracking URI do DagsHub: {e}")
    elif tracking_uri_env:
        try:
            # Se não houver owner/repo, usar exatamente o URI fornecido por env
            mlflow.set_tracking_uri(tracking_uri_env)
        except Exception as e:
            warnings.warn(f"Falha ao definir MLFLOW_TRACKING_URI: {e}")

    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "max_receita_cafeterias")
    try:
        mlflow.set_experiment(exp_name)
    except Exception as e:
        warnings.warn(f"Falha ao definir experimento '{exp_name}' no MLflow: {e}. Usando experimento padrão.")

    if dagshub is not None and owner and repo:
        try:
            dagshub.init(
                repo_owner=owner,
                repo_name=repo,
                mlflow=True,
            )
        except Exception as e:
            warnings.warn(f"Falha ao iniciar DagsHub: {e}")


def load_dataset() -> pd.DataFrame:
    """
    Carrega o dataset principal; se não existir (CI/ambiente limpo), gera um dataset sintético
    determinístico para permitir treino e testes estáveis.
    """
    if DADOS_AMOR_A_CAKES.exists():
        df = pd.read_excel(DADOS_AMOR_A_CAKES)
    else:
        rng = np.random.default_rng(42)
        n = 600
        dates = pd.date_range("2021-07-01", periods=n, freq="D")
        preco_original = rng.uniform(20.0, 60.0, size=n)
        desconto_pct = rng.uniform(0.0, 0.04, size=n)
        custo_producao = rng.uniform(5.0, 15.0, size=n)
        preco_final = preco_original * (1.0 - desconto_pct)
        # Relação: demanda cai com preço, sobe com desconto; ruído controlado
        base_qty = 500 - 3.5 * preco_final + 1200 * desconto_pct
        qty_noise = rng.normal(0.0, 25.0, size=n)
        quantidade_vendida_mes = np.clip(base_qty + qty_noise, a_min=0, a_max=None)
        df = pd.DataFrame({
            'data': dates,
            'custo_producao': custo_producao,
            'preco_original': preco_original,
            'desconto_pct': desconto_pct,
            'preco_final': preco_final,
            'quantidade_vendida_mes': quantidade_vendida_mes,
        })

    df = df.drop_duplicates()
    if 'data' in df.columns:
        df['data'] = pd.to_datetime(df['data'], errors='coerce')
    if {'preco_final','quantidade_vendida_dia'}.issubset(df.columns):
        df['receita_dia'] = df['preco_final'] * df['quantidade_vendida_dia']
    if {'preco_final','quantidade_vendida_mes'}.issubset(df.columns):
        df['receita_mes'] = df['preco_final'] * df['quantidade_vendida_mes']
    return df


def select_features(df: pd.DataFrame):
    target = 'quantidade_vendida_mes' if 'quantidade_vendida_mes' in df.columns else 'quantidade_vendida_dia'
    feature_cols = [c for c in ['custo_producao','preco_original','desconto_pct','preco_final'] if c in df.columns]
    X = df[feature_cols].copy()
    y = df[target].copy()
    return X, y, feature_cols, target


def get_model_specs():
    specs = [
        ModelSpec("LinearRegression", LinearRegression(), {}),
        ModelSpec("RidgeCV", Pipeline([('scaler', StandardScaler()), ('model', RidgeCV(alphas=np.logspace(-3, 3, 21), cv=5))]), {}),
        ModelSpec("LassoCV", Pipeline([('scaler', StandardScaler()), ('model', LassoCV(alphas=None, cv=5, max_iter=5000))]), {}),
        ModelSpec("ElasticNetCV", Pipeline([('scaler', StandardScaler()), ('model', ElasticNetCV(l1_ratio=[.1,.3,.5,.7,.9], cv=5, max_iter=5000))]), {}),
        ModelSpec("RandomForest", RandomForestRegressor(n_estimators=300, random_state=42), {}),
        ModelSpec("GradientBoosting", GradientBoostingRegressor(random_state=42), {}),
    ]
    if XGBRegressor is not None:
        specs.append(ModelSpec("XGBRegressor", XGBRegressor(random_state=42, n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.9, colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=1.0), {}))
    return specs


def _simulate_expected_profit(model, X_valid: pd.DataFrame) -> float:
    """Simula lucro esperado por linha ao buscar o melhor desconto até 4%.
    Retorna média do lucro máximo por linha na validação.
    """
    if not {'preco_original','custo_producao'}.issubset(X_valid.columns):
        return float('nan')
    grid = np.arange(0.0, 0.0400001, 0.001)
    best_profit = np.full(shape=(len(X_valid),), fill_value=-np.inf, dtype=float)
    for d in grid:
        Xd = X_valid.copy()
        if 'desconto_pct' in Xd.columns:
            Xd['desconto_pct'] = d
        if 'preco_final' in Xd.columns:
            Xd['preco_final'] = X_valid['preco_original'] * (1.0 - d)
        preds = np.asarray(model.predict(Xd), dtype=float)
        preco_final = X_valid['preco_original'] * (1.0 - d)
        profit_d = (preco_final - X_valid['custo_producao']) * preds
        best_profit = np.maximum(best_profit, profit_d.values if hasattr(profit_d, 'values') else profit_d)
    return float(np.nanmean(best_profit))


def cross_validate_model(model, X: pd.DataFrame, y: pd.Series, cv):
    rmses, maes, r2s, profits = [], [], [], []
    for train_idx, valid_idx in cv.split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        # Compatibilidade com versões mais antigas do scikit-learn onde 'squared' não existe
        try:
            rmse = mean_squared_error(y_valid, y_pred, squared=False)
        except TypeError:
            rmse = float(np.sqrt(mean_squared_error(y_valid, y_pred)))
        rmses.append(rmse)
        maes.append(mean_absolute_error(y_valid, y_pred))
        r2s.append(r2_score(y_valid, y_pred))
        profits.append(_simulate_expected_profit(model, X_valid))
    return {
        'cv_rmse_mean': float(np.mean(rmses)), 'cv_rmse_std': float(np.std(rmses)),
        'cv_mae_mean': float(np.mean(maes)), 'cv_mae_std': float(np.std(maes)),
        'cv_r2_mean': float(np.mean(r2s)), 'cv_r2_std': float(np.std(r2s)),
        'cv_expected_profit_mean': float(np.mean(profits)), 'cv_expected_profit_std': float(np.std(profits)),
    }


def export_linear_json(model, feature_cols, target, out_path: Path):
    intercept = getattr(model, 'intercept_', 0.0)
    coefs = getattr(model, 'coef_', np.zeros(len(feature_cols)))
    payload = {
        'model_type': type(model).__name__,
        'intercept': float(intercept),
        'coefficients': {c: float(v) for c, v in zip(feature_cols, coefs)},
        'target': target,
        'features': feature_cols,
        'preprocessing': {'scaling': 'none', 'derived': ['receita_dia','receita_mes']},
        'inference_notes': 'Compute preco_final = preco_original * (1 - desconto_pct) before prediction.'
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def main():
    init_tracking()
    df = load_dataset()
    X, y, feature_cols, target = select_features(df)
    # Validação temporal quando existir coluna de data
    if 'data' in df.columns and pd.api.types.is_datetime64_any_dtype(df['data']):
        # ordenar por tempo e sincronizar X/y
        order = np.argsort(df['data'].fillna(pd.Timestamp.min).values)
        X = X.iloc[order].reset_index(drop=True)
        y = y.iloc[order].reset_index(drop=True)
        cv = TimeSeriesSplit(n_splits=5)
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    best_profit = -float('inf')
    best_entry = None

    for spec in get_model_specs():
        with mlflow.start_run(run_name=f"{spec.name}"):
            mlflow.log_params({'model': spec.name, **spec.params})
            metrics = cross_validate_model(spec.estimator, X, y, cv)
            # Aliases compatíveis com testes: cv_rmse, cv_mae, cv_r2
            metrics_with_alias = {
                **metrics,
                'cv_rmse': metrics.get('cv_rmse_mean'),
                'cv_mae': metrics.get('cv_mae_mean'),
                'cv_r2': metrics.get('cv_r2_mean'),
            }
            mlflow.log_metrics(metrics_with_alias)

            # Fit/holdout metrics
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            spec.estimator.fit(X_train, y_train)
            y_pred = spec.estimator.predict(X_test)
            try:
                rmse_holdout = mean_squared_error(y_test, y_pred, squared=False)
            except TypeError:
                rmse_holdout = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            holdout = {
                'rmse': float(rmse_holdout),
                'r2': float(r2_score(y_test, y_pred)),
                'mae': float(mean_absolute_error(y_test, y_pred)),
            }
            mlflow.log_metrics({f"holdout_{k}": v for k, v in holdout.items()})

            # Persist model locally
            local_model_path = MODELS_DIR / f"{spec.name}.pkl"
            try:
                import joblib
                joblib.dump(spec.estimator, local_model_path)
                mlflow.log_artifact(str(local_model_path))
            except Exception:
                pass

            entry = {
                'name': spec.name,
                'metrics': {**metrics_with_alias, **{f"holdout_{k}": v for k, v in holdout.items()}},
                'model_path': str(local_model_path),
                'estimator': spec.estimator,
            }
            results.append(entry)

            # Seleção por lucro esperado médio em validação
            if metrics.get('cv_expected_profit_mean', -float('inf')) > best_profit:
                best_profit = metrics['cv_expected_profit_mean']
                best_entry = entry

            mlflow.set_tag('selection_metric', 'cv_expected_profit_mean')

    assert best_entry is not None, "Nenhum modelo foi avaliado."

    # Salvar melhor modelo
    best_label = best_entry['name']
    best_model = best_entry['estimator']
    best_out = MODELS_DIR / 'best_model_max_receita.pkl'
    import joblib
    joblib.dump(best_model, best_out)

    # Exportar JSON para UI: se melhor for linear, exporta; caso contrário, exporta LinearRegression como baseline
    json_target_path = MODELS_DIR / 'model_linear.json'
    if isinstance(best_model, (LinearRegression, RidgeCV, LassoCV, ElasticNetCV)):
        export_linear_json(best_model, feature_cols, target, json_target_path)
    else:
        # Treinar baseline linear para inferência no front-end
        baseline = LinearRegression().fit(X, y)
        export_linear_json(baseline, feature_cols, target, json_target_path)

    # Artefato: curva média de receita/lucro vs desconto
    try:
        discounts = np.arange(0.0, 0.0400001, 0.001)
        rows = []
        for d in discounts:
            Xd = X.copy()
            if 'desconto_pct' in Xd.columns:
                Xd['desconto_pct'] = d
            if 'preco_final' in Xd.columns:
                Xd['preco_final'] = X['preco_original'] * (1.0 - d)
            qty = np.asarray(best_model.predict(Xd), dtype=float)
            price = X['preco_original'] * (1.0 - d)
            revenue = price * qty
            profit = (price - X['custo_producao']) * qty
            rows.append({
                'desconto_pct': d,
                'mean_quantity': float(np.mean(qty)),
                'mean_revenue': float(np.mean(revenue)),
                'mean_profit': float(np.mean(profit)),
            })
        curve_path = MODELS_DIR / 'curve_business_metric.csv'
        pd.DataFrame(rows).to_csv(curve_path, index=False)
        mlflow.log_artifact(str(curve_path))
    except Exception as e:
        warnings.warn(f"Falha ao gerar curva de negócio: {e}")

    # Métrica do melhor modelo
    best_metrics = best_entry['metrics']
    # Snapshot de métricas para testes (consumido por tests/expected_metrics.json)
    try:
        snapshot = {
            'best_model': best_label,
            'metrics': best_metrics,
        }
        snapshot_path = MODELS_DIR / 'metrics_snapshot.json'
        snapshot_path.write_text(json.dumps(snapshot, indent=2), encoding='utf-8')
        try:
            mlflow.log_artifact(str(snapshot_path))
        except Exception:
            pass
    except Exception as e:
        warnings.warn(f"Falha ao salvar metrics_snapshot.json: {e}")
    print(f"Melhor modelo: {best_label}\nMétricas: {best_metrics}\nExport JSON: {json_target_path}")

    # SHAP: explicabilidade para modelos de árvore; para lineares já há coeficientes
    try:
        explainer = None
        model_for_shap = best_model
        if isinstance(best_model, Pipeline):
            # tenta obter o estimador final
            try:
                model_for_shap = best_model.named_steps.get('model', best_model)
            except Exception:
                model_for_shap = best_model
        if HAS_SHAP and shap is not None and (isinstance(model_for_shap, (RandomForestRegressor, GradientBoostingRegressor)) or type(model_for_shap).__name__ == 'XGBRegressor'):
            # Amostra para desempenho
            X_sample = X.sample(n=min(1000, len(X)), random_state=42)
            # Usa os dados no espaço original; alguns pipelines podem exigir transform
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_sample)
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, plot_type='bar', show=False)
            shap_path = MODELS_DIR / 'shap_summary.png'
            plt.tight_layout()
            plt.savefig(shap_path, dpi=140)
            plt.close()
            mlflow.log_artifact(str(shap_path))
    except Exception as e:
        warnings.warn(f"Falha ao gerar SHAP: {e}")


if __name__ == '__main__':
    main()