import numpy as np
import pandas as pd
import pytest
from src.data.loader import compute_log_returns, create_lagged_features


def test_compute_log_returns():
    prices = [100.0, 105.0, 102.0, 110.0]
    df = pd.DataFrame({"Price": prices})
    res = compute_log_returns(df)

    assert len(res) == 3
    assert "log_return" in res.columns
    expected_first = np.log(105.0 / 100.0)
    assert np.isclose(res["log_return"].iloc[0], expected_first)


def test_create_lagged_features():
    returns = np.array([0.01, -0.02, 0.03, 0.015, -0.005])
    n_lags = 2
    X, y, _ = create_lagged_features(returns, n_lags=n_lags)

    assert X.shape == (3, 2)
    assert len(y) == 3
    # Le premier target y[0] doit correspondre à returns[2] = 0.03
    assert np.isclose(y[0], 0.03)
    # Ses features X[0] doivent être [returns[0], returns[1]]
    assert np.allclose(X[0], [0.01, -0.02])
