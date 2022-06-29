import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from src.config.paths import DADOS_AMOR_A_CAKES, MODELS_DIR


def load_dataset() -> pd.DataFrame:
    df = pd.read_excel(DADOS_AMOR_A_CAKES)
    # Base cleaning aligned with EDA
    df = df.drop_duplicates()
    if 'data' in df.columns:
        df['data'] = pd.to_datetime(df['data'], errors='coerce')
    # Auxiliary revenue columns
    if {'preco_final','quantidade_vendida_dia'}.issubset(df.columns):
        df['receita_dia'] = df['preco_final'] * df['quantidade_vendida_dia']
    if {'preco_final','quantidade_vendida_mes'}.issubset(df.columns):
        df['receita_mes'] = df['preco_final'] * df['quantidade_vendida_mes']
    return df


def select_features(df: pd.DataFrame):
    target = (
        'quantidade_vendida_mes'
        if 'quantidade_vendida_mes' in df.columns
        else 'quantidade_vendida_dia'
    )
    feature_cols = [
        c
        for c in ['custo_producao', 'preco_original', 'desconto_pct', 'preco_final']
        if c in df.columns
    ]
    X = df[feature_cols].copy()
    y = df[target].copy()
    return X, y, feature_cols, target


def train_and_evaluate(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    # CV scores for robustness
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rmse = -cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=cv)
    cv_r2 = cross_val_score(model, X, y, scoring='r2', cv=cv)

    metrics = {
        'rmse': float(rmse),
        'r2': float(r2),
        'cv_rmse_mean': float(np.mean(cv_rmse)),
        'cv_rmse_std': float(np.std(cv_rmse)),
        'cv_r2_mean': float(np.mean(cv_r2)),
        'cv_r2_std': float(np.std(cv_r2)),
    }
    return model, metrics


def export_model_json(model: LinearRegression, feature_cols, target: str, out_path: Path):
    payload = {
        'model_type': 'LinearRegression',
        'intercept': float(model.intercept_),
        'coefficients': {col: float(coef) for col, coef in zip(feature_cols, model.coef_)},
        'target': target,
        'features': feature_cols,
        'preprocessing': {
            'scaling': 'none',
            'derived': ['receita_dia', 'receita_mes'],
        },
        'inference_notes': (
            'Compute preco_final = preco_original * (1 - desconto_pct) before prediction.'
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def main():
    print(f"Carregando dataset: {DADOS_AMOR_A_CAKES}")
    df = load_dataset()
    X, y, feature_cols, target = select_features(df)
    print(f"Features: {feature_cols}; Target: {target}; Shape: {X.shape}")
    model, metrics = train_and_evaluate(X, y)
    print("Métricas:", metrics)

    json_path = MODELS_DIR / 'model_linear.json'
    export_model_json(model, feature_cols, target, json_path)
    print(f"Modelo exportado para: {json_path}")


if __name__ == '__main__':
    main()