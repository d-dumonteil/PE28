"""
MLP-ARCH Portfolio Dashboard
Streamlit app for backtesting & visualizing MLP-ARCH volatility models.

Run with:
    pip install streamlit yfinance torch scipy plotly
    streamlit run dashboard.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize
from typing import Callable, List, Dict, Tuple

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MLP-ARCH Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252836);
        border: 1px solid #2e3250;
        border-radius: 12px;
        padding: 1rem 1.4rem;
        text-align: center;
    }
    .metric-label { color: #8890b5; font-size: 0.78rem; font-weight: 600;
                    letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 4px; }
    .metric-value { color: #e8ecff; font-size: 1.6rem; font-weight: 700; }
    .metric-value.positive { color: #4ade80; }
    .metric-value.negative { color: #f87171; }
    .section-header { color: #a5b4fc; font-size: 0.9rem; font-weight: 700;
                      text-transform: uppercase; letter-spacing: 0.08em;
                      margin-bottom: 0.6rem; border-bottom: 1px solid #2e3250;
                      padding-bottom: 0.4rem; }
    div[data-testid="stTabs"] button { color: #8890b5 !important; }
    div[data-testid="stTabs"] button[aria-selected="true"] { color: #a5b4fc !important; }
    .stProgress > div > div > div { background-color: #6366f1; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODEL DEFINITIONS (inlined)
# ─────────────────────────────────────────────

class MLPVolatilityModelBasic(nn.Module):
    """AR(1)-MLP-ARCH (sans volume)"""
    def __init__(self, k: int, q: int, activation: Callable = torch.relu):
        super().__init__()
        self.k = k
        self.q = q
        self.activation = activation
        self.a0 = nn.Parameter(torch.tensor(0.0))
        self.a1 = nn.Parameter(torch.tensor(0.0))
        self.raw_beta0 = nn.Parameter(torch.tensor(-4.0))
        self.raw_beta = nn.Parameter(-4.0 + 0.1 * torch.randn(k))
        self.raw_w0 = nn.Parameter(-8.0 + 0.1 * torch.randn(k))
        self.raw_w = nn.Parameter(-4.0 + 0.1 * torch.randn(k, q))

    def sigma2(self, u2_lags: torch.Tensor) -> torch.Tensor:
        w0 = torch.nn.functional.softplus(self.raw_w0)
        w = torch.nn.functional.softplus(self.raw_w)
        lin = w0.unsqueeze(0) + u2_lags @ w.T
        h = self.activation(lin)
        beta0 = torch.nn.functional.softplus(self.raw_beta0)
        beta = torch.nn.functional.softplus(self.raw_beta)
        return beta0 + (h * beta.unsqueeze(0)).sum(dim=1)

    def negative_log_likelihood(self, r: torch.Tensor, init_u2=None) -> torch.Tensor:
        T = r.shape[0]
        r_lag = r[:-1]; r_curr = r[1:]
        u = r_curr - self.a0 - self.a1 * r_lag
        u2 = u**2
        if init_u2 is None:
            init_u2 = torch.var(u2).repeat(self.q)
        u2_full = torch.cat([init_u2, u2], dim=0)
        u2_lags = []
        for t in range(self.q, self.q + T - 1):
            u2_lags.append(u2_full[t - self.q:t].flip(0))
        u2_lags = torch.stack(u2_lags, dim=0)
        sigma2 = self.sigma2(u2_lags) + 1e-8
        n = u.shape[0]
        nll = (0.5 * n * torch.log(torch.tensor(2.0 * torch.pi))
               + 0.5 * torch.sum(torch.log(sigma2))
               + 0.5 * torch.sum(u**2 / sigma2))
        return nll

    def predict_next(self, r_hist: np.ndarray, vol_hist=None) -> Tuple[float, float]:
        r = torch.tensor(r_hist, dtype=torch.float32)
        mu_next = self.a0.item() + self.a1.item() * r[-1].item()
        r_lag = r[:-1]; r_curr = r[1:]
        u = r_curr - self.a0 - self.a1 * r_lag
        u2 = u**2
        u2_lags = u2[-self.q:].flip(0).unsqueeze(0)
        sigma2_next = self.sigma2(u2_lags).item()
        return mu_next, np.sqrt(max(sigma2_next, 1e-10))


class MLPVolatilityModelImproved(nn.Module):
    """MLP-ARCH amélioré avec volume et terme GARCH"""
    def __init__(self, k: int, q: int, activation: Callable = torch.relu):
        super().__init__()
        self.k = k; self.q = q; self.activation = activation
        self.a0 = nn.Parameter(torch.tensor(0.0))
        self.a1 = nn.Parameter(torch.tensor(0.0))
        self.raw_beta0 = nn.Parameter(torch.tensor(-4.0))
        self.raw_beta = nn.Parameter(-4.0 + 0.1 * torch.randn(k))
        self.raw_w0 = nn.Parameter(-8.0 + 0.1 * torch.randn(k))
        self.raw_w = nn.Parameter(-4.0 + 0.1 * torch.randn(k, q))
        self.raw_gamma = nn.Parameter(-4.0 + 0.1 * torch.randn(k, q))
        self.raw_rho = nn.Parameter(-4.0 + 0.1 * torch.randn(k))

    def get_params(self):
        return (torch.nn.functional.softplus(self.raw_beta0),
                torch.nn.functional.softplus(self.raw_beta),
                torch.nn.functional.softplus(self.raw_w0),
                torch.nn.functional.softplus(self.raw_w),
                torch.nn.functional.softplus(self.raw_gamma),
                torch.nn.functional.softplus(self.raw_rho))

    def negative_log_likelihood(self, r: torch.Tensor, volume: torch.Tensor, init_u2=None) -> torch.Tensor:
        T = r.shape[0]
        r_lag = r[:-1]; r_curr = r[1:]
        u = r_curr - self.a0 - self.a1 * r_lag
        u2 = u**2
        beta0, beta, w0, w, gamma, rho = self.get_params()
        if init_u2 is None:
            init_u2 = torch.var(u2).detach().repeat(self.q)
        u2_full = torch.cat([init_u2, u2], dim=0)
        v_init = volume[0].repeat(self.q)
        vol_full = torch.cat([v_init, volume[:-1]], dim=0)
        u2_windows = u2_full.unfold(0, self.q, 1).flip(1)
        vol_windows = vol_full.unfold(0, self.q, 1).flip(1)
        static_input = w0.unsqueeze(0) + (u2_windows @ w.T) + (vol_windows @ gamma.T)
        sigmas2 = []
        sigma2_prev = torch.var(u2).detach() + 1e-6
        for t in range(static_input.shape[0]):
            lin = static_input[t] + rho * sigma2_prev
            h = self.activation(lin)
            val = beta0 + (h * beta).sum() + 1e-8
            sigmas2.append(val); sigma2_prev = val
        sigmas2 = torch.stack(sigmas2)
        n = static_input.shape[0]
        nll = (0.5 * n * torch.log(torch.tensor(2.0 * torch.pi))
               + 0.5 * torch.sum(torch.log(sigmas2))
               + 0.5 * torch.sum(u**2 / sigmas2))
        return nll

    def predict_next(self, r_hist: np.ndarray, vol_hist: np.ndarray) -> Tuple[float, float]:
        r = torch.tensor(r_hist, dtype=torch.float32)
        vol = torch.tensor(vol_hist, dtype=torch.float32)
        mu_next = self.a0.item() + self.a1.item() * r[-1].item()
        self.eval()
        with torch.no_grad():
            r_lag = r[:-1]; r_curr = r[1:]
            u = r_curr - self.a0 - self.a1 * r_lag
            u2 = u**2
            init_val = torch.var(u2)
            init_u2 = init_val.repeat(self.q)
            u2_cat = torch.cat([init_u2, u2], dim=0)
            v_init = vol[0].repeat(self.q)
            vol_cat = torch.cat([v_init, vol], dim=0)
            beta0, beta, w0, w, gamma, rho = self.get_params()
            mu_last = self.a0 + self.a1 * r[-2]
            u_last = r[-1] - mu_last
            u2_extended = torch.cat([u2_cat, u_last.view(1)**2])
            u2_windows = u2_extended.unfold(0, self.q, 1).flip(1)
            vol_windows = vol_cat.unfold(0, self.q, 1).flip(1)
            static_input = w0.unsqueeze(0) + (u2_windows @ w.T) + (vol_windows @ gamma.T)
            curr_sigma2 = init_val + 1e-6
            for t in range(static_input.shape[0]):
                lin = static_input[t] + rho * curr_sigma2
                h = self.activation(lin)
                curr_sigma2 = beta0 + (h * beta).sum() + 1e-8
        return mu_next, np.sqrt(max(curr_sigma2.item(), 1e-10))


# ─────────────────────────────────────────────
# TRAINING FUNCTION
# ─────────────────────────────────────────────
def fit_model(r_train, vol_train, r_val, vol_val,
              model_type, k, q, activation, lr, epochs,
              progress_bar=None, status_text=None):
    ModelClass = MLPVolatilityModelImproved if model_type == "Amélioré (+ Volume)" else MLPVolatilityModelBasic
    r_t = torch.tensor(r_train, dtype=torch.float32)
    v_t = torch.tensor(vol_train, dtype=torch.float32) if vol_train is not None else None
    r_v = torch.tensor(r_val, dtype=torch.float32) if r_val is not None else None
    v_v = torch.tensor(vol_val, dtype=torch.float32) if vol_val is not None else None

    model = ModelClass(k=k, q=q, activation=activation)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        if model_type == "Amélioré (+ Volume)":
            loss = model.negative_log_likelihood(r_t, v_t)
        else:
            loss = model.negative_log_likelihood(r_t)
        loss.backward(); optimizer.step()
        history["train_loss"].append(float(loss.item()))

        if r_v is not None:
            model.eval()
            with torch.no_grad():
                if model_type == "Amélioré (+ Volume)":
                    val_loss = model.negative_log_likelihood(r_v, v_v)
                else:
                    val_loss = model.negative_log_likelihood(r_v)
            history["val_loss"].append(float(val_loss.item()))
        else:
            history["val_loss"].append(np.nan)

        if progress_bar is not None and epoch % max(1, epochs // 100) == 0:
            progress_bar.progress(epoch / epochs)
        if status_text is not None and epoch % max(1, epochs // 10) == 0:
            vl = history["val_loss"][-1]
            vl_str = f"{vl:.4f}" if not np.isnan(vl) else "—"
            status_text.text(f"Epoch {epoch}/{epochs} | Train NLL: {loss.item():.4f} | Val NLL: {vl_str}")

    return model, history


# ─────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────
def download_data(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    try:
        if isinstance(df.columns, pd.MultiIndex):
            lvl0 = df.columns.get_level_values(0).unique().tolist()
            if 'Close' in lvl0:
                prices = df['Close']; volumes = df['Volume']
            else:
                df2 = df.swaplevel(0, 1, axis=1)
                prices = df2['Close']; volumes = df2['Volume']
        else:
            prices = df[['Close']]; volumes = df[['Volume']]
            prices.columns = tickers; volumes.columns = tickers
    except Exception:
        prices = df['Close']; volumes = df['Volume']
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
        volumes = volumes.to_frame(tickers[0])
    return prices.dropna(how="all"), volumes.dropna(how="all")


def compute_log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()


def prepare_volume(volumes):
    v_log = np.log(volumes.replace(0, np.nan).ffill().fillna(1.0) + 1.0)
    mean_vol = v_log.mean()
    if isinstance(mean_vol, pd.Series):
        mean_vol[mean_vol < 1e-6] = 1.0
    return v_log / mean_vol


def optimal_weights(mu, cov, ridge=1e-6):
    n = len(mu)
    cov_reg = cov + ridge * np.eye(n)
    def objective(w):
        pv = np.sqrt(w @ cov_reg @ w)
        return 1e10 if pv < 1e-10 else -(w @ mu) / pv
    def c_l1(w): return 1.0 - np.sum(np.abs(w))
    try:
        w0 = np.linalg.inv(cov_reg) @ mu
        w0 /= (np.sum(np.abs(w0)) + 1e-10)
    except Exception:
        w0 = np.ones(n) / n
    res = minimize(objective, w0, method='SLSQP',
                   constraints={'type': 'ineq', 'fun': c_l1},
                   options={'maxiter': 1000, 'ftol': 1e-9})
    return res.x if res.success else w0


def compute_metrics(returns):
    r = pd.Series(returns)
    cumret = (1 + r).cumprod()
    total_ret = cumret.iloc[-1] - 1
    ann_ret = (1 + total_ret) ** (252 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    roll_max = cumret.cummax()
    dd = (cumret - roll_max) / roll_max
    max_dd = dd.min()
    win_rate = (r > 0).mean()
    return dict(total_ret=total_ret, ann_ret=ann_ret, ann_vol=ann_vol,
                sharpe=sharpe, max_dd=max_dd, win_rate=win_rate)


ACTIVATION_MAP = {
    "ReLU": torch.relu,
    "Tanh": torch.tanh,
    "Sigmoid": torch.sigmoid,
    "ELU": torch.nn.functional.elu,
}

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    st.markdown('<p class="section-header">Univers</p>', unsafe_allow_html=True)
    default_tickers = "AAPL, MSFT, GOOGL, AMZN"
    tickers_input = st.text_input("Tickers (séparés par des virgules)", value=default_tickers)
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Début", value=pd.Timestamp("2010-01-01"))
    with col2:
        end_date = st.date_input("Fin", value=pd.Timestamp("2024-12-31"))

    st.markdown("---")
    st.markdown('<p class="section-header">Modèle</p>', unsafe_allow_html=True)
    model_type = st.selectbox("Variante", ["Amélioré (+ Volume)", "Basique (AR-MLP-ARCH)"])
    activation_name = st.selectbox("Activation", list(ACTIVATION_MAP.keys()))
    activation_fn = ACTIVATION_MAP[activation_name]

    k = st.slider("Neurones cachés (k)", 2, 30, 10)
    q = st.slider("Lags (q)", 1, 10, 3)
    lr = st.select_slider("Learning rate", options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
                          value=1e-2, format_func=lambda x: f"{x:.0e}")
    epochs = st.slider("Epochs", 100, 2000, 400, step=50)

    st.markdown("---")
    st.markdown('<p class="section-header">Split (années)</p>', unsafe_allow_html=True)
    train_years = st.slider("Train", 1, 12, 8)
    val_years = st.slider("Validation", 1, 3, 1)
    test_years = st.slider("Test", 1, 3, 1)

    st.markdown("---")
    run_btn = st.button("🚀 Lancer le Backtest", use_container_width=True, type="primary")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:1.2rem">
  <div style="font-size:2rem">📈</div>
  <div>
    <div style="font-size:1.5rem;font-weight:800;color:#e8ecff;letter-spacing:-0.02em">
      MLP-ARCH Portfolio Dashboard</div>
    <div style="color:#8890b5;font-size:0.85rem">
      Modèle de volatilité MLP-ARCH · Optimisation de portefeuille Max-Sharpe</div>
  </div>
</div>
""", unsafe_allow_html=True)

tab_data, tab_train, tab_backtest, tab_analysis = st.tabs(
    ["📊 Données", "🧠 Entraînement", "💼 Backtest", "🔬 Analyse"])

# ─────────────────────────────────────────────
# TAB 1 – DATA
# ─────────────────────────────────────────────
with tab_data:
    st.markdown("### Chargement & exploration des données")

    if not tickers:
        st.warning("Entrez au moins un ticker dans la barre latérale.")
    else:
        with st.spinner("Téléchargement des données…"):
            try:
                prices, volumes = download_data(tickers, str(start_date), str(end_date))
                rets = compute_log_returns(prices)
                st.session_state["prices"] = prices
                st.session_state["volumes"] = volumes
                st.session_state["rets"] = rets
            except Exception as e:
                st.error(f"Erreur de téléchargement : {e}")
                prices = None

        if prices is not None and not prices.empty:
            # Stats rapides
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Tickers</div>'
                            f'<div class="metric-value">{len(prices.columns)}</div></div>',
                            unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Observations</div>'
                            f'<div class="metric-value">{len(prices):,}</div></div>',
                            unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Début</div>'
                            f'<div class="metric-value" style="font-size:1rem">{prices.index[0].date()}</div></div>',
                            unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Fin</div>'
                            f'<div class="metric-value" style="font-size:1rem">{prices.index[-1].date()}</div></div>',
                            unsafe_allow_html=True)

            st.markdown("")
            col_l, col_r = st.columns(2)

            # Prix normalisés
            with col_l:
                st.markdown("**Prix normalisés (base 100)**")
                fig = go.Figure()
                colors = px.colors.qualitative.Vivid
                for i, col in enumerate(prices.columns):
                    norm = prices[col] / prices[col].iloc[0] * 100
                    fig.add_trace(go.Scatter(
                        x=prices.index, y=norm, name=col,
                        line=dict(color=colors[i % len(colors)], width=1.8)))
                fig.update_layout(
                    template="plotly_dark", plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                    margin=dict(l=0, r=0, t=10, b=0), height=320,
                    legend=dict(orientation="h", y=1.05),
                    xaxis=dict(gridcolor="#1e2130"), yaxis=dict(gridcolor="#1e2130"))
                st.plotly_chart(fig, use_container_width=True)

            # Rendements rolling vol
            with col_r:
                st.markdown("**Volatilité réalisée (fenêtre 21j)**")
                fig = go.Figure()
                for i, col in enumerate(rets.columns):
                    rv = rets[col].rolling(21).std() * np.sqrt(252) * 100
                    fig.add_trace(go.Scatter(
                        x=rv.index, y=rv, name=col,
                        line=dict(color=colors[i % len(colors)], width=1.8)))
                fig.update_layout(
                    template="plotly_dark", plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                    margin=dict(l=0, r=0, t=10, b=0), height=320,
                    yaxis_title="Vol annualisée (%)",
                    legend=dict(orientation="h", y=1.05),
                    xaxis=dict(gridcolor="#1e2130"), yaxis=dict(gridcolor="#1e2130"))
                st.plotly_chart(fig, use_container_width=True)

            # Matrice de corrélation
            st.markdown("**Matrice de corrélation des rendements**")
            corr = rets.corr()
            fig_corr = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.columns,
                colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                text=np.round(corr.values, 2), texttemplate="%{text}",
                showscale=True))
            fig_corr.update_layout(
                template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                margin=dict(l=0, r=0, t=10, b=0), height=350)
            st.plotly_chart(fig_corr, use_container_width=True)

# ─────────────────────────────────────────────
# TAB 2 – TRAINING
# ─────────────────────────────────────────────
with tab_train:
    st.markdown("### Entraînement du modèle")

    if run_btn:
        if "prices" not in st.session_state:
            st.warning("Allez d'abord sur l'onglet **Données** pour charger les données.")
        else:
            prices = st.session_state["prices"]
            volumes = st.session_state["volumes"]
            rets = st.session_state["rets"]
            vols_scaled = prepare_volume(volumes).loc[rets.index]

            days = 252
            n_train = train_years * days
            n_val = val_years * days
            n_test = test_years * days
            total_needed = n_train + n_val + n_test

            if len(rets) < total_needed:
                st.error(f"Pas assez de données ({len(rets)} jours disponibles, {total_needed} requis). "
                         "Élargissez la plage de dates ou réduisez les horizons.")
            else:
                train = rets.iloc[:n_train]
                val = rets.iloc[n_train:n_train + n_val]
                test = rets.iloc[n_train + n_val:n_train + n_val + n_test]

                v_train = vols_scaled.iloc[:n_train]
                v_val = vols_scaled.iloc[n_train:n_train + n_val]

                corr = rets.iloc[:n_train + n_val].corr().values

                trained_models = {}
                all_histories = {}

                st.markdown(f"**Entraînement de {len(tickers)} modèle(s) — {epochs} epochs**")
                overall_bar = st.progress(0)
                overall_status = st.empty()

                loss_placeholder = st.empty()

                for i_tick, ticker in enumerate(rets.columns):
                    overall_status.text(f"Modèle {i_tick+1}/{len(rets.columns)} : {ticker}")
                    tick_bar = st.progress(0)
                    tick_status = st.empty()

                    model, hist = fit_model(
                        r_train=train[ticker].values,
                        vol_train=(v_train[ticker].values
                                   if model_type == "Amélioré (+ Volume)" else None),
                        r_val=val[ticker].values,
                        vol_val=(v_val[ticker].values
                                 if model_type == "Amélioré (+ Volume)" else None),
                        model_type=model_type,
                        k=k, q=q, activation=activation_fn,
                        lr=lr, epochs=epochs,
                        progress_bar=tick_bar,
                        status_text=tick_status,
                    )
                    trained_models[ticker] = model
                    all_histories[ticker] = hist
                    tick_bar.progress(1.0)
                    tick_status.text(f"✅ {ticker} terminé — NLL finale : {hist['train_loss'][-1]:.4f}")
                    overall_bar.progress((i_tick + 1) / len(rets.columns))

                overall_status.text("✅ Entraînement terminé !")

                st.session_state["models"] = trained_models
                st.session_state["histories"] = all_histories
                st.session_state["train"] = train
                st.session_state["val"] = val
                st.session_state["test"] = test
                st.session_state["v_scaled"] = vols_scaled
                st.session_state["corr"] = corr
                st.session_state["n_train"] = n_train
                st.session_state["n_val"] = n_val
                st.session_state["model_type"] = model_type

                # Loss curves
                st.markdown("#### Courbes de perte (moyennées sur tous les tickers)")
                avg_train = np.nanmean([all_histories[t]["train_loss"] for t in rets.columns], axis=0)
                avg_val = np.nanmean([all_histories[t]["val_loss"] for t in rets.columns], axis=0)
                epochs_arr = np.arange(1, epochs + 1)

                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(x=epochs_arr, y=avg_train,
                                              name="Train NLL", line=dict(color="#6366f1", width=2)))
                fig_loss.add_trace(go.Scatter(x=epochs_arr, y=avg_val,
                                              name="Val NLL", line=dict(color="#f59e0b", width=2, dash="dash")))
                fig_loss.update_layout(
                    template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                    xaxis_title="Epoch", yaxis_title="NLL",
                    margin=dict(l=0, r=0, t=20, b=0), height=350,
                    legend=dict(orientation="h", y=1.05),
                    xaxis=dict(gridcolor="#1e2130"), yaxis=dict(gridcolor="#1e2130"))
                st.plotly_chart(fig_loss, use_container_width=True)

    elif "histories" in st.session_state:
        all_histories = st.session_state["histories"]
        rets_cols = list(all_histories.keys())
        _ep = epochs
        avg_train = np.nanmean([all_histories[t]["train_loss"] for t in rets_cols], axis=0)
        avg_val = np.nanmean([all_histories[t]["val_loss"] for t in rets_cols], axis=0)
        epochs_arr = np.arange(1, len(avg_train) + 1)

        st.markdown("#### Courbes de perte (dernière exécution)")
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=epochs_arr, y=avg_train,
                                      name="Train NLL", line=dict(color="#6366f1", width=2)))
        fig_loss.add_trace(go.Scatter(x=epochs_arr, y=avg_val,
                                      name="Val NLL", line=dict(color="#f59e0b", width=2, dash="dash")))
        fig_loss.update_layout(
            template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            xaxis_title="Epoch", yaxis_title="NLL",
            margin=dict(l=0, r=0, t=20, b=0), height=380,
            legend=dict(orientation="h", y=1.05),
            xaxis=dict(gridcolor="#1e2130"), yaxis=dict(gridcolor="#1e2130"))
        st.plotly_chart(fig_loss, use_container_width=True)

        # Per-ticker selector
        sel_tick = st.selectbox("Courbe par ticker", rets_cols)
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=epochs_arr, y=all_histories[sel_tick]["train_loss"],
                                   name="Train", line=dict(color="#4ade80", width=2)))
        fig_t.add_trace(go.Scatter(x=epochs_arr, y=all_histories[sel_tick]["val_loss"],
                                   name="Val", line=dict(color="#fb923c", width=2, dash="dash")))
        fig_t.update_layout(
            template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            xaxis_title="Epoch", yaxis_title="NLL", title=sel_tick,
            margin=dict(l=0, r=0, t=40, b=0), height=320,
            xaxis=dict(gridcolor="#1e2130"), yaxis=dict(gridcolor="#1e2130"))
        st.plotly_chart(fig_t, use_container_width=True)
    else:
        st.info("Configurez les paramètres dans la barre latérale puis cliquez sur **🚀 Lancer le Backtest**.")

# ─────────────────────────────────────────────
# TAB 3 – BACKTEST RESULTS
# ─────────────────────────────────────────────
with tab_backtest:
    st.markdown("### Résultats du Backtest")

    def run_backtest_inference(trained_models, rets, vols_scaled, corr,
                               n_train, n_val, test, model_type_sel):
        results = []
        pbar = st.progress(0)
        status = st.empty()
        n_days = len(test)

        for t in range(n_days):
            date = test.index[t]
            mu_pred, vol_pred = [], []
            for ticker in rets.columns:
                hist_r = rets[ticker].iloc[:n_train + n_val + t].values
                hist_v = (vols_scaled[ticker].iloc[:n_train + n_val + t].values
                          if model_type_sel == "Amélioré (+ Volume)" else None)
                mu_t, vol_t = trained_models[ticker].predict_next(hist_r, hist_v)
                mu_pred.append(mu_t); vol_pred.append(vol_t)

            mu_pred = np.array(mu_pred); vol_pred = np.array(vol_pred)
            cov_t = np.outer(vol_pred, vol_pred) * corr
            w = optimal_weights(mu_pred, cov_t)
            invested = np.sum(np.abs(w))
            cash = 1.0 - invested
            r_real = test.iloc[t].values
            port_ret = np.dot(w, r_real) + cash * 0.0

            results.append({
                "date": date, "port_return": port_ret,
                "invested_fraction": invested, "cash_fraction": cash,
                **{f"mu_{c}": mu_pred[i] for i, c in enumerate(rets.columns)},
                **{f"vol_{c}": vol_pred[i] for i, c in enumerate(rets.columns)},
                **{f"w_{c}": w[i] for i, c in enumerate(rets.columns)},
            })
            if t % max(1, n_days // 50) == 0:
                pbar.progress((t + 1) / n_days)
                status.text(f"Inférence jour {t+1}/{n_days}…")

        pbar.progress(1.0); status.text("✅ Inférence terminée !")
        return pd.DataFrame(results).set_index("date")

    # Run inference if backtest button pressed
    if run_btn and "models" in st.session_state:
        with st.spinner("Inférence sur la période de test…"):
            df_res = run_backtest_inference(
                st.session_state["models"],
                st.session_state["rets"],
                st.session_state["v_scaled"],
                st.session_state["corr"],
                st.session_state["n_train"],
                st.session_state["n_val"],
                st.session_state["test"],
                st.session_state["model_type"],
            )
            st.session_state["results"] = df_res

    if "results" not in st.session_state:
        st.info("Lancez le backtest depuis la barre latérale.")
    else:
        df_res = st.session_state["results"]
        test = st.session_state["test"]
        rets_cols = list(st.session_state["models"].keys())
        colors = px.colors.qualitative.Vivid

        # ── Metrics ──
        m = compute_metrics(df_res["port_return"])
        bm_returns = test.mean(axis=1)
        bm = compute_metrics(bm_returns)

        st.markdown("#### Métriques de performance (période de test)")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        def mc(label, val, fmt=".2%", good_positive=True):
            cls = ("positive" if val > 0 else "negative") if good_positive else (
                  "negative" if val > 0 else "positive")
            return (f'<div class="metric-card"><div class="metric-label">{label}</div>'
                    f'<div class="metric-value {cls}">{val:{fmt}}</div></div>')
        with c1: st.markdown(mc("Rendement Total", m["total_ret"]), unsafe_allow_html=True)
        with c2: st.markdown(mc("Rendement Ann.", m["ann_ret"]), unsafe_allow_html=True)
        with c3: st.markdown(mc("Vol. Ann.", m["ann_vol"], good_positive=False), unsafe_allow_html=True)
        with c4: st.markdown(mc("Sharpe", m["sharpe"], ".2f"), unsafe_allow_html=True)
        with c5: st.markdown(mc("Max Drawdown", m["max_dd"], good_positive=False), unsafe_allow_html=True)
        with c6: st.markdown(mc("Win Rate", m["win_rate"], ".1%"), unsafe_allow_html=True)

        st.markdown("")

        # ── Cumulative Returns ──
        cum_port = (1 + df_res["port_return"]).cumprod()
        cum_bm = (1 + bm_returns).cumprod()

        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(x=cum_port.index, y=cum_port.values * 100 - 100,
                                     name="Portefeuille MLP-ARCH",
                                     line=dict(color="#6366f1", width=2.5),
                                     fill="tozeroy", fillcolor="rgba(99,102,241,0.08)"))
        fig_cum.add_trace(go.Scatter(x=cum_bm.index, y=cum_bm.values * 100 - 100,
                                     name="Benchmark (EW)",
                                     line=dict(color="#f59e0b", width=1.8, dash="dash")))
        fig_cum.add_hline(y=0, line=dict(color="#4b5563", width=1, dash="dot"))
        fig_cum.update_layout(
            template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            title="Rendement cumulé (%)", yaxis_title="Rendement (%)",
            margin=dict(l=0, r=0, t=40, b=0), height=380,
            legend=dict(orientation="h", y=1.08),
            xaxis=dict(gridcolor="#1e2130"), yaxis=dict(gridcolor="#1e2130"))
        st.plotly_chart(fig_cum, use_container_width=True)

        col_a, col_b = st.columns(2)

        # ── Drawdown ──
        with col_a:
            roll_max = cum_port.cummax()
            dd = (cum_port - roll_max) / roll_max * 100
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=dd.index, y=dd.values, name="Drawdown",
                                        line=dict(color="#f87171", width=1.5),
                                        fill="tozeroy", fillcolor="rgba(248,113,113,0.15)"))
            fig_dd.update_layout(
                template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                title="Drawdown (%)", margin=dict(l=0, r=0, t=40, b=0), height=280,
                xaxis=dict(gridcolor="#1e2130"), yaxis=dict(gridcolor="#1e2130"))
            st.plotly_chart(fig_dd, use_container_width=True)

        # ── Fraction investie ──
        with col_b:
            fig_inv = go.Figure()
            fig_inv.add_trace(go.Scatter(x=df_res.index, y=df_res["invested_fraction"] * 100,
                                         name="Investie", line=dict(color="#4ade80", width=1.5),
                                         fill="tozeroy", fillcolor="rgba(74,222,128,0.1)"))
            fig_inv.add_trace(go.Scatter(x=df_res.index, y=df_res["cash_fraction"] * 100,
                                         name="Cash", line=dict(color="#60a5fa", width=1.5, dash="dash")))
            fig_inv.update_layout(
                template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                title="Fraction investie vs Cash (%)",
                margin=dict(l=0, r=0, t=40, b=0), height=280,
                legend=dict(orientation="h", y=1.1),
                xaxis=dict(gridcolor="#1e2130"), yaxis=dict(gridcolor="#1e2130"))
            st.plotly_chart(fig_inv, use_container_width=True)

        # ── Poids ──
        st.markdown("#### Évolution des poids de portefeuille")
        w_cols = [c for c in df_res.columns if c.startswith("w_")]
        fig_w = go.Figure()
        for i, wc in enumerate(w_cols):
            ticker = wc[2:]
            fig_w.add_trace(go.Scatter(x=df_res.index, y=df_res[wc],
                                       name=ticker, stackgroup="one",
                                       line=dict(width=0),
                                       fillcolor=colors[i % len(colors)]))
        fig_w.update_layout(
            template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            margin=dict(l=0, r=0, t=10, b=0), height=320,
            legend=dict(orientation="h", y=1.05),
            xaxis=dict(gridcolor="#1e2130"), yaxis=dict(gridcolor="#1e2130"))
        st.plotly_chart(fig_w, use_container_width=True)

        # ── Daily Returns Distribution ──
        st.markdown("#### Distribution des rendements journaliers")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=df_res["port_return"] * 100, name="Portefeuille",
            nbinsx=60, marker_color="#6366f1", opacity=0.75))
        fig_hist.add_trace(go.Histogram(
            x=bm_returns * 100, name="Benchmark EW",
            nbinsx=60, marker_color="#f59e0b", opacity=0.55))
        fig_hist.update_layout(
            barmode="overlay",
            template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            xaxis_title="Rendement journalier (%)",
            margin=dict(l=0, r=0, t=10, b=0), height=300,
            legend=dict(orientation="h", y=1.05),
            xaxis=dict(gridcolor="#1e2130"), yaxis=dict(gridcolor="#1e2130"))
        st.plotly_chart(fig_hist, use_container_width=True)

        # ── Export ──
        st.markdown("#### Export")
        csv = df_res.reset_index().to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Télécharger les résultats (.csv)", data=csv,
                           file_name="mlp_arch_backtest.csv", mime="text/csv")

# ─────────────────────────────────────────────
# TAB 4 – ANALYSIS
# ─────────────────────────────────────────────
with tab_analysis:
    st.markdown("### Analyse de la volatilité prédite")

    if "results" not in st.session_state:
        st.info("Lancez le backtest d'abord.")
    else:
        df_res = st.session_state["results"]
        test = st.session_state["test"]
        rets_cols = list(st.session_state["models"].keys())
        colors = px.colors.qualitative.Vivid

        sel_ticker = st.selectbox("Ticker à analyser", rets_cols)

        vol_col = f"vol_{sel_ticker}"
        mu_col = f"mu_{sel_ticker}"

        if vol_col in df_res.columns:
            col_l2, col_r2 = st.columns(2)

            with col_l2:
                # Predicted vol vs realized vol
                realized_vol = test[sel_ticker].rolling(5).std() * np.sqrt(252) * 100
                pred_vol = df_res[vol_col] * np.sqrt(252) * 100

                fig_v = go.Figure()
                fig_v.add_trace(go.Scatter(x=pred_vol.index, y=pred_vol.values,
                                           name="Vol prédite (ann.)",
                                           line=dict(color="#6366f1", width=2)))
                fig_v.add_trace(go.Scatter(x=realized_vol.index, y=realized_vol.values,
                                           name="Vol réalisée 5j",
                                           line=dict(color="#f59e0b", width=1.5, dash="dash")))
                fig_v.update_layout(
                    template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                    title=f"Volatilité {sel_ticker} — prédite vs réalisée (%)",
                    margin=dict(l=0, r=0, t=40, b=0), height=320,
                    legend=dict(orientation="h", y=1.1),
                    xaxis=dict(gridcolor="#1e2130"), yaxis=dict(gridcolor="#1e2130"))
                st.plotly_chart(fig_v, use_container_width=True)

            with col_r2:
                # Predicted mu
                fig_mu = make_subplots(specs=[[{"secondary_y": True}]])
                fig_mu.add_trace(go.Bar(x=test.index, y=test[sel_ticker] * 100,
                                        name="Rendement réel",
                                        marker_color=np.where(test[sel_ticker] >= 0, "#4ade80", "#f87171")),
                                 secondary_y=False)
                fig_mu.add_trace(go.Scatter(x=df_res.index, y=df_res[mu_col] * 100,
                                            name="μ prédit", mode="lines",
                                            line=dict(color="#c084fc", width=1.5)),
                                 secondary_y=True)
                fig_mu.update_layout(
                    template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                    title=f"Rendements {sel_ticker} — réels vs μ prédit (%)",
                    margin=dict(l=0, r=0, t=40, b=0), height=320,
                    legend=dict(orientation="h", y=1.1),
                    xaxis=dict(gridcolor="#1e2130"), yaxis=dict(gridcolor="#1e2130"))
                st.plotly_chart(fig_mu, use_container_width=True)

            # Scatter: predicted vol vs |realized return|
            st.markdown(f"**Scatter : σ prédite vs |rendement réel| ({sel_ticker})**")
            abs_r = test[sel_ticker].reindex(df_res.index).abs() * 100
            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(
                x=df_res[vol_col] * 100, y=abs_r,
                mode="markers",
                marker=dict(color="#6366f1", size=5, opacity=0.5),
                name=sel_ticker))
            fig_sc.update_layout(
                template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                xaxis_title="σ prédite (%)", yaxis_title="|Rendement réel| (%)",
                margin=dict(l=0, r=0, t=10, b=0), height=320,
                xaxis=dict(gridcolor="#1e2130"), yaxis=dict(gridcolor="#1e2130"))
            st.plotly_chart(fig_sc, use_container_width=True)

        # ── Comparaison Sharpe par ticker ──
        st.markdown("#### Sharpe individuel (position longue pure, test)")
        sharpe_vals = {}
        for tick in rets_cols:
            r = test[tick]
            s = np.sqrt(252) * r.mean() / r.std(ddof=0) if r.std(ddof=0) > 0 else 0
            sharpe_vals[tick] = s

        fig_sh = go.Figure(go.Bar(
            x=list(sharpe_vals.keys()),
            y=list(sharpe_vals.values()),
            marker_color=[("#4ade80" if v > 0 else "#f87171") for v in sharpe_vals.values()]))
        fig_sh.update_layout(
            template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            yaxis_title="Sharpe annualisé",
            margin=dict(l=0, r=0, t=10, b=0), height=300,
            xaxis=dict(gridcolor="#1e2130"), yaxis=dict(gridcolor="#1e2130"))
        st.plotly_chart(fig_sh, use_container_width=True)
