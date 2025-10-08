import pandas as pd

from src.config.paths import DADOS_AMOR_A_CAKES
from src.train import load_and_prepare_data, train_and_evaluate


def test_training_pipeline_runs():
    df = load_and_prepare_data(DADOS_AMOR_A_CAKES)
    assert isinstance(df, pd.DataFrame)
    model, metrics = train_and_evaluate(df)
    assert "rmse" in metrics and "r2" in metrics
    # Métricas devem ser números finitos
    assert metrics["rmse"] >= 0
    assert -1 <= metrics["r2"] <= 1