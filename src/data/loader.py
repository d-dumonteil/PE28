"""Data loading, cleaning, and preprocessing utilities for financial time series."""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
import yfinance as yf


def load_price_data(
    source: Union[str, Path],
    ticker: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    price_col: str = "Close",
) -> pd.DataFrame:
    """Load asset price and volume data from yfinance or local files (csv, txt, xlsx).

    Parameters
    ----------
    source : str or Path
        Either 'yfinance' or a path to a local file.
    ticker : str, optional
        Ticker symbol if downloading via yfinance.
    start_date : str, optional
        Start date in 'YYYY-MM-DD' format.
    end_date : str, optional
        End date in 'YYYY-MM-DD' format.
    price_col : str, default 'Close'
        Column name to consider as primary price.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by Datetime with standard columns: Price, Volume.
    """
    if str(source).lower() == "yfinance":
        if not ticker:
            raise ValueError("A ticker must be provided when source='yfinance'.")
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[[price_col, "Volume"]].rename(columns={price_col: "Price"})
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        if suffix in [".csv", ".txt"]:
            # Detect separator (comma, semicolon, or whitespace)
            df = pd.read_csv(path, sep=None, engine="python")
        elif suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        # Attempt standard date column parsing
        date_candidates = ["Date", "date", "Timestamp", "timestamp", "Time", "time"]
        date_col = next((c for c in date_candidates if c in df.columns), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col).sort_index()

        target_price = next(
            (c for c in [price_col, "Price", "Close", "Adj Close", "Dernier", "Clôture"] if c in df.columns),
            None,
        )
        if not target_price:
            raise KeyError(f"Could not identify a price column among: {list(df.columns)}")

        target_vol = next((c for c in ["Volume", "Vol.", "volume"] if c in df.columns), None)
        cols_map = {target_price: "Price"}
        if target_vol:
            cols_map[target_vol] = "Volume"

        df = df[list(cols_map.keys())].rename(columns=cols_map)

    # Clean & ensure numeric types
    df = df.dropna().astype(float)
    return df


def compute_log_returns(
    df: pd.DataFrame, price_col: str = "Price", dropna: bool = True
) -> pd.DataFrame:
    """Compute log-returns r_t = ln(P_t / P_{t-1}) and normalized log-volume."""
    data = df.copy()
    data["log_return"] = np.log(data[price_col] / data[price_col].shift(1))

    if "Volume" in data.columns:
        # Standardized log-volume for volume-augmented models
        log_vol = np.log1p(data["Volume"])
        data["log_volume"] = (log_vol - log_vol.mean()) / (log_vol.std() + 1e-8)

    if dropna:
        data = data.dropna()
    return data


def create_lagged_features(
    series: np.ndarray,
    n_lags: int,
    volume_series: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Build lagged feature matrices for autoregressive / ARCH neural models.

    Parameters
    ----------
    series : np.ndarray
        Target return series of shape (T,).
    n_lags : int
        Number of previous time steps (lags) to include.
    volume_series : np.ndarray, optional
        Normalized volume series of shape (T,).

    Returns
    -------
    X_returns : np.ndarray of shape (T - n_lags, n_lags)
    y : np.ndarray of shape (T - n_lags,)
    X_volume : np.ndarray of shape (T - n_lags, n_lags) or None
    """
    T = len(series)
    if T <= n_lags:
        raise ValueError(f"Series length {T} must be greater than n_lags {n_lags}")

    X_ret = np.column_stack([series[i : T - n_lags + i] for i in range(n_lags)])
    y = series[n_lags:]

    X_vol = None
    if volume_series is not None:
        X_vol = np.column_stack(
            [volume_series[i : T - n_lags + i] for i in range(n_lags)]
        )

    return X_ret, y, X_vol
