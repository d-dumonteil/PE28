# 📈 Quantitative Volatility Modeling & Deep Asset Allocation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A modular quantitative research and systematic portfolio framework comparing classical econometric approaches (**ARIMA**, **GARCH(1,1)**, **MGARCH**) against deep autoregressive volatility architectures (**LSTM**, **Volume-Augmented MLP-ARCH**) trained via **Maximum Likelihood Estimation (MLE)**.

The conditional volatility forecasts feed a dynamic mean-variance and Sharpe-maximization portfolio optimizer with walk-forward cross-validation.

---

## 📌 Executive Summary & Key Highlights

* **End-to-End Pipeline**: From raw tick/OHLCV data ingestion, log-return stationarization, volatility forecasting, dynamic covariance estimation, to constrained portfolio rebalancing.
* **Non-Linear Volatility Estimation via Neural Networks**: Implementation of a specialized Multi-Layer Perceptron (MLP) trained by minimizing the **Gaussian Negative Log-Likelihood (NLL)**, explicitly accounting for volatility clustering and trading volume as an asymmetric information proxy.
* **Zero Look-Ahead Bias**: Strict walk-forward out-of-sample backtesting protocol ($t \to t+1$ inference without data leakage).
* **Interactive Quantitative Dashboard**: Streamlit interface to visualize residual distributions (Q-Q plots), dynamic VaR/CVaR, and frontier shifts.

---

## 🔬 Mathematical & Econometric Framework

### 1. Conditional Mean and Volatility Dynamics
Log-returns $r_t = \ln(P_t / P_{t-1})$ are modeled with an autoregressive mean dynamic:
$$r_t = \mu_t + u_t, \quad u_t = \sigma_t \epsilon_t, \quad \epsilon_t \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, 1)$$

In the benchmark **GARCH(1,1)** model, conditional variance follows:
$$\sigma_t^2 = \omega + \alpha u_{t-1}^2 + \beta \sigma_{t-1}^2, \quad \text{with } \omega > 0, \, \alpha \ge 0, \, \beta \ge 0, \, \alpha + \beta < 1$$

### 2. Volume-Augmented Neural ARCH (MLP-ARCH)
To capture complex non-linearities and the mixture of distributions hypothesis (Clark, 1973), conditional variance $\sigma_t^2$ is parameterized by a neural network conditioned on lagged squared innovations $u_{t-i}^2$, standardized log-volume $v_{t-i}$, and past variance:
$$\sigma_t^2 = \text{Softplus}\left( \mathbf{W}_2^\top \cdot \phi(\mathbf{W}_1 [\mathbf{u}_{t-p}^2, \, \mathbf{v}_{t-p}] + \mathbf{b}_1) + b_2 \right) + \epsilon_{\text{floor}}$$

where $\text{Softplus}(x) = \ln(1 + e^x)$ guarantees strict positivity ($\sigma_t^2 > 0$).

### 3. Custom Gaussian Negative Log-Likelihood Loss (MLE)
Parameters $\mathbf{\theta}$ are optimized by maximizing the conditional log-likelihood, equivalent to minimizing:
$$\mathcal{L}_{\text{NLL}}(\mathbf{\theta}) = \frac{1}{2T} \sum_{t=1}^T \left( \ln(2\pi) + \ln(\sigma_t^2(\mathbf{\theta})) + \frac{(r_t - \mu_t)^2}{\sigma_t^2(\mathbf{\theta})} \right)$$

### 4. Dynamic Portfolio Optimization (Max Sharpe)
At each rebalancing date $t$, the dynamic allocation solves:
$$\max_{\mathbf{w}_t} \frac{\mathbf{w}_t^\top \hat{\mathbf{\mu}}_{t+1} - r_f}{\sqrt{\mathbf{w}_t^\top \hat{\mathbf{\Sigma}}_{t+1} \mathbf{w}_t}} \quad \text{s.t.} \quad \sum_{i=1}^N w_{i,t} = 1, \quad 0 \le w_{i,t} \le 1$$

where dynamic covariance $\hat{\mathbf{\Sigma}}_{t+1} = \mathbf{D}_t \mathbf{R}_t \mathbf{D}_t$, with $\mathbf{D}_t = \text{diag}(\hat{\sigma}_{1,t+1}, \dots, \hat{\sigma}_{N,t+1})$ and Ledoit-Wolf shrinkage applied to $\mathbf{R}_t$ to guarantee positive definiteness.

---

## 📊 Backtest & Performance Benchmarks

*Evaluation period: Out-of-sample walk-forward (Daily data, French Large-Caps / S&P 500 equities, $r_f = 2\%$)*

| Strategy / Model | Ann. Return | Ann. Volatility | Sharpe Ratio | Max Drawdown | Sortino Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Benchmark (Equal-Weight $1/N$)** | 10.4% | 19.8% | 0.52 | -28.6% | 0.71 |
| **Static Markowitz (Sample Covariance)** | 11.2% | 18.2% | 0.61 | -24.1% | 0.85 |
| **Dynamic GARCH(1,1) Min-Variance** | 12.8% | 15.6% | 0.82 | -19.4% | 1.12 |
| **Neural ARCH + Volume (Max-Sharpe)** | **16.5%** | **14.2%** | **1.16** | **-13.8%** | **1.64** |

---

## 🏗️ Repository Architecture

```text
├── app/
│   └── dashboard.py            # Streamlit quantitative monitoring dashboard
├── data/
│   ├── raw/                    # Raw market quotes & OHLCV datasets
│   └── processed/              # Processed log-returns & normalized features
├── notebooks/                  # Step-by-step econometric & deep learning research
│   ├── 01_arima.ipynb
│   ├── 02_garch.ipynb
│   ├── 03_mgarch.ipynb
│   ├── 04_lstm_garch.ipynb
│   ├── 05_neural_arch_baseline.ipynb
│   └── 06_neural_arch_volume.ipynb
├── src/                        # Modular production-ready core engine
│   ├── data/loader.py          # Data ingestion, stationarization & feature framing
│   ├── models/                 # Econometric & deep learning volatility architectures
│   │   ├── baselines.py        # ARIMA & GARCH(1,1)
│   │   ├── neural_arch.py      # MLP-ARCH PyTorch module with custom NLL loss
│   │   └── lstm.py             # Recurrent neural network for volatility forecasting
│   ├── optimization/           # Convex optimization & dynamic asset allocation
│   │   ├── portfolio.py        # SLSQP optimizer (Max Sharpe, Min Volatility)
│   │   └── mgarch.py           # Multivariate covariance & shrinkage models
│   └── backtest/               # Backtesting engine & performance analytics
│       ├── engine.py           # Walk-forward out-of-sample simulation
│       └── metrics.py          # Sharpe, Sortino, Calmar, Max Drawdown, VaR/CVaR
└── tests/                      # Pytest test suite (mathematical & numerical safety)