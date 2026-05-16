# 🏦 NSE PSU Banking Stock Direction Predictor

> **A systematic ML study across 10 Indian Public Sector Undertaking (PSU) banking stocks. Ten independent direction-prediction models were built and evaluated; Bank of Maharashtra (`MAHABANK.NS`) emerged as the best-performing model and is published here in its final form (V9 Balanced).**

---

## 🗂️ Table of Contents

- [The Study](#the-study)
- [Why PSU Banks?](#why-psu-banks)
- [The 10 Models](#the-10-models)
- [Why BOM Won](#why-bom-won)
- [BOM Model Architecture](#bom-model-architecture)
- [Feature Engineering](#feature-engineering)
- [Training Pipeline](#training-pipeline)
- [Backtest Setup](#backtest-setup)
- [Key Design Decisions (V9 Improvements)](#key-design-decisions-v9-improvements)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Outputs](#outputs)
- [Disclaimer](#disclaimer)

---

## The Study

This project is the result of building **10 independent ML models** — one per stock — across the PSU banking sector of the NSE. Each model follows an identical pipeline:

- Same feature engineering logic (55+ technical indicators)
- Same three-way temporal split (Train 2010–2019 / Val 2020–2021 / Test 2022–2025)
- Same model architecture (Ensemble: Random Forest + HistGradientBoosting)
- Same evaluation framework (classification metrics + expanding-window backtest)
- Same realistic cost model (full NSE delivery charges, ~0.30–0.35% round-trip)

The models were compared on **out-of-sample test set performance** (2022–2025). The **Bank of Maharashtra (BOM)** model delivered the strongest and most consistent results across accuracy, F1, Sharpe ratio, and drawdown — and is the model published in this repository.

---

## Why PSU Banks?

Indian PSU (government-owned) banks form a distinct, correlated cluster within the Nifty Bank index. They share common macro drivers — RBI policy, government recapitalisation decisions, credit cycle exposure, and election-year budgetary changes — making them an interesting and cohesive universe for a cross-stock ML study.

All 10 stocks are listed on the **National Stock Exchange (NSE)** and are components of the **Nifty PSU Bank Index**.

---

## The 10 Models

| # | Bank | NSE Ticker | Notes |
|---|---|---|---|
| 1 | State Bank of India | `SBIN.NS` | Largest PSU bank by assets |
| 2 | UCO Bank | `UCOBANK.NS` | Mid-size; high beta to sector |
| 3 | Bank of Baroda | `BANKBARODA.NS` | Large-cap, internationally active |
| 4 | Canara Bank | `CANBK.NS` | Large-cap; strong retail base |
| 5 | **Bank of Maharashtra** ⭐ | **`MAHABANK.NS`** | **Best model — published here** |
| 6 | Punjab National Bank | `PNB.NS` | Large-cap; high NPA history |
| 7 | Central Bank of India | `CENTRALBK.NS` | Mid-size; government-focused |
| 8 | Union Bank of India | `UNIONBANK.NS` | Post-amalgamation entity |
| 9 | Indian Overseas Bank | `IOB.NS` | High retail exposure |
| 10 | Bank of India | `BANKINDIA.NS` | Mid-to-large cap |

>

---

## Why BOM Won

Bank of Maharashtra stood out across the evaluation criteria for several reasons:

- **Predictable price behaviour:** BOM's daily price action showed stronger autocorrelation patterns in the technical indicators used as features, making next-day direction more learnable than noisier large-caps like SBI or PNB.
- **Consistent test accuracy:** The model maintained above-50% accuracy across all four test years (2022–2025) in the walk-forward evaluation — something few of the 10 models achieved uniformly.
- **Strong backtest risk metrics:** BOM's ensemble delivered a favourable Sharpe ratio and contained maximum drawdown compared to the Buy & Hold baseline — after applying full NSE transaction costs.
- **Lower noise-to-signal ratio:** Smaller-cap PSU stocks like BOM tend to be less driven by large institutional block trades and index-rebalancing, making technical indicators relatively more informative.
- **Sector feature synergy:** BOM's returns showed a consistent and learnable relative-strength relationship with the Nifty Bank index, which the `BOM_vs_NiftyBank` feature captures effectively.

---

## BOM Model Architecture

### Final Model: Ensemble (Random Forest + HistGradientBoosting)

The final production model is a **soft-voting ensemble** that averages the predicted probabilities of two complementary base classifiers:

```
┌──────────────────────────────────────────────────────────────────────┐
│                    ENSEMBLE MODEL (RF + HGB)                         │
│                                                                      │
│  Input: FINAL_features (importance-pruned via MDI)                   │
│                                                                      │
│  ┌───────────────────────────┐    ┌──────────────────────────────┐   │
│  │   Random Forest (RF)      │    │  HistGradientBoosting (HGB)  │   │
│  │   class_weight='balanced' │    │  max_iter=200                │   │
│  │   oob_score=True          │    │  learning_rate=0.01          │   │
│  │   bootstrap=True          │    │  max_depth=4                 │   │
│  │   TimeSeriesSplit CV k=5  │    │  min_samples_leaf=80         │   │
│  │   RandomizedSearchCV      │    │  l2_regularization=5.0       │   │
│  │   scoring='f1'            │    │  native NaN support          │   │
│  └─────────────┬─────────────┘    └──────────────┬───────────────┘   │
│                │                                 │                   │
│                └────────────┬────────────────────┘                   │
│                             ▼                                        │
│                    Average Probabilities                             │
│                             │                                        │
│                             ▼                                        │
│             Threshold  (tuned on VAL set only, then frozen)          │
│                             │                                        │
│              ┌──────────────┴──────────────┐                         │
│              ▼                             ▼                         │
│         Signal = 1 (BUY)            Signal = 0 (HOLD / SELL)        │
└──────────────────────────────────────────────────────────────────────┘
```

### Base Classifier Comparison

| Property | Random Forest | HistGradientBoosting |
|---|---|---|
| Type | Bagging (parallel trees) | Boosting (sequential trees) |
| Missing values | Requires imputation | Native support |
| Regularization | `class_weight`, `min_samples_leaf` | `l2_regularization`, `min_samples_leaf` |
| Hyperparameter search | `RandomizedSearchCV` (30 iter) | Fixed architecture |
| CV strategy | `TimeSeriesSplit` k=5 | — |

### RF Hyperparameter Search Space

```python
param_dist = {
    'n_estimators'     : [100, 300, 500, 800],
    'max_depth'        : [3, 6, 12, None],
    'min_samples_leaf' : [10, 30, 50],
    'min_samples_split': [20, 50, 100],
    'max_features'     : ['sqrt', 'log2'],
    'bootstrap'        : [True],
}
```

---

## Feature Engineering

The model builds **55+ engineered features** entirely from backward-looking data (no look-ahead). Features are grouped into 10 categories:

### 1. Core Price & Volume
| Feature | Description |
|---|---|
| `Daily_Return_%` | Day-over-day % change in Adj Close |
| `Daily_Range` | High − Low |
| `Price_change` | Adj Close − Open |
| `Volume_Ratio` | Volume ÷ 20-day average volume |

### 2. Trend (Moving Averages & Crossovers)
| Feature | Description |
|---|---|
| `MA20`, `MA50`, `MA200` | Simple moving averages |
| `MA20_above_MA50` | Binary MA crossover signal |
| `Price_above_MA50` | Binary price position signal |
| `Price_above_MA200` | Binary price position signal |

### 3. Momentum
| Feature | Description |
|---|---|
| `MACD`, `MACD_Hist` | MACD (EMA12 − EMA26) and histogram |
| `RSI`, `RSI_norm` | 14-day RSI and normalised (÷100) |

### 4. Volatility
| Feature | Description |
|---|---|
| `STD20` | 20-day rolling standard deviation |
| `BB_pct` | Bollinger Band % position |
| `ATR14` | 14-day Average True Range |

### 5. Volume & Flow (Stationary)
| Feature | Description |
|---|---|
| `OBV_ROC` | OBV 20-day rate-of-change *(not cumulative)* |
| `PVT_ROC` | PVT 20-day rate-of-change *(not cumulative)* |
| `Vol_SMA_Ratio` | Volume ÷ 5-day average volume |
| `Vol_Momentum` | Volume pct_change |
| `Vol_Shock` | Binary: volume > 2× 10-day average |
| `MFI` | Money Flow Index (14-day) |
| `CMF` | Chaikin Money Flow (20-day) |

### 6. Institutional Benchmark (Rolling VWAP)
| Feature | Description |
|---|---|
| `VWAP` | 20-day rolling VWAP *(not cumulative since 2010)* |
| `VWAP_Ratio` | Close ÷ rolling VWAP |

### 7. Trend Strength (ADX — Wilder's EMA)
| Feature | Description |
|---|---|
| `ADX` | Average Directional Index (14-day) |
| `+DI`, `-DI` | Positive and Negative Directional Indicators |

### 8. Lag Features (1, 2, 3, 5 days)
`RSI_Lag{n}` · `Return_Lag{n}` · `MACD_Lag{n}` · `Volume_Lag{n}`

### 9. Nifty Bank Sector Features
| Feature | Description |
|---|---|
| `NiftyBank_Lag1` | Prior-day Nifty Bank return |
| `NiftyBank_Trend` | MA5 > MA20 binary signal on sector index |
| `BOM_vs_NiftyBank` | BOM daily return minus Nifty Bank return |

### 10. Calendar
| Feature | Description |
|---|---|
| `DayOfWeek` | 0 = Monday … 4 = Friday |

---

## Training Pipeline

```
Raw OHLCV Data — MAHABANK.NS + ^NSEBANK (2010–2025, via yfinance)
                           │
                           ▼
          Feature Engineering (55+ features)
                           │
                           ▼
          Three-Way Temporal Split
   ┌─────────────────────────────────────────────┐
   │  TRAIN      2010–2019  (RF & HGB weights)   │
   │  VALIDATION 2020–2021  (threshold tuning)   │
   │  TEST       2022–2025  (final eval — once)  │
   └─────────────────────────────────────────────┘
                           │
                           ▼
   Stage 1 — Full-Feature RF (all 55+ features)
                           │
                           ▼
   Stage 2 — MDI Importance Pruning
             cutoff = mean importance
             keeps only features with importance ≥ mean
                           │
                           ▼
   Stage 3 — Retrain RF on impactful_features
             + RandomizedSearchCV (30 iter, TimeSeriesSplit k=5, F1)
   +
   Train HGB on same impactful_features
                           │
                           ▼
   Ensemble: Average(RF_proba, HGB_proba)
                           │
                           ▼
   Threshold Search on VALIDATION SET only
   (best F1 subject to coverage ≥ 15%)
                           │
                           ▼
   ★ Freeze Threshold → Evaluate on TEST SET once
```

### Expanding-Window Backtest

The full ensemble is **retrained every year** on all available history to prevent staleness:

| Predict Year | Train Window | Validation Window |
|---|---|---|
| 2022 | 2010–2019 | 2020–2021 |
| 2023 | 2010–2020 | 2021–2022 |
| 2024 | 2010–2021 | 2022–2023 |
| 2025 | 2010–2022 | 2023–2024 |

---

## Backtest Setup

The backtest simulates a **long-only, daily-signal strategy** on `MAHABANK.NS` from 2022 to 2025, compared against Buy & Hold and a Random Baseline.

### Full NSE Delivery Transaction Cost Model

| Charge | Rate | Applied On |
|---|---|---|
| Brokerage | min(0.1%, ₹20) | Both legs |
| STT | 0.1% | Sell leg only |
| Stamp Duty | 0.015% | Buy leg only |
| Exchange Fee | 0.00335% | Both legs |
| SEBI Charge | 0.0001% | Both legs |
| GST on brokerage | 18% of brokerage | Both legs |
| **Total round-trip** | **~0.30–0.35%** | |

> This is approximately **3× more expensive** than a brokerage-only model — making the backtest significantly more conservative and realistic for Indian equity delivery trades.

**Metrics reported:** Final Capital · Total Return % · CAGR % · Sharpe Ratio · Max Drawdown % · Calmar Ratio

---

## Key Design Decisions (V9 Improvements)

V9 resolves all known sources of **optimistic bias** from earlier versions of all 10 models:

| # | Change | Old Behaviour | New Behaviour |
|---|---|---|---|
| 1 | OBV / PVT | Cumulative from 2010 — unbounded, causes train/test distribution drift | 20-day rate-of-change — stationary |
| 2 | VWAP | Cumulative since 2010 — essentially the 15-year avg price by 2024 | 20-day rolling VWAP |
| 3 | Data split | 2-way (train / test) | 3-way: Train 2010–2019 / Val 2020–2021 / Test 2022–2025 |
| 4 | Threshold tuning | Tuned on test set → inflated accuracy by 1–4 pp | Tuned on validation set only, frozen before test |
| 5 | Transaction costs | Brokerage only (~0.1% round-trip) | Full NSE statutory charges (~0.30–0.35% round-trip) |
| 6 | Model retraining | Static — trained once in 2021, predict 4 years frozen | Expanding window — retrained every year |
| 7 | CSV export | No ground-truth column | `Target_Return_Fwd` + `Correct_Prediction` + `Model_Type` |
| 8 | Calmar ratio | Not reported | Added to metrics table |
| 9 | Feature selection | Manual domain-intuition drops | Data-driven MDI importance pruning (cutoff = mean) |
| 10 | Model consistency | Three disconnects: HGB dead code; backtest used wrong features; ensemble never reached backtest | Single pipeline: `FINAL_features`, `FINAL_thresh`, `FINAL_model_type` propagate to all sections |

---

## Project Structure

```
📁 NSE-PSU-Banking-Stock-Predictor/
│
├── BOM_V9_balanced.ipynb               ← Champion model notebook (Bank of Maharashtra)
├── MAHABANK_V9_backtest_daily.csv      ← Daily signals, capital curve, predictions (generated)
├── MAHABANK_V9_backtest_metrics.csv    ← Summary metrics table (generated)
└── README.md                           ← This file
```

> The other 9 model notebooks (SBI, UCO, BOB, Canara, PNB, Central Bank, Union Bank, IOB, Bank of India) followed the same pipeline but are not published here, as BOM was selected as the best-performing model from the study.

---

## Requirements

```
python >= 3.9
pandas
numpy
yfinance
scikit-learn
matplotlib
seaborn
```

Install all at once:
```bash
pip install pandas numpy yfinance scikit-learn matplotlib seaborn
```

---

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/rakmo-n/NSE-PSU-Banking-Stock-Direction-Predictor.git
   cd NSE-PSU-Banking-Stock-Direction-Predictor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Open the notebook**
   ```bash
   jupyter notebook BOM_V9_balanced.ipynb
   ```

4. **Run all cells top-to-bottom.** The notebook will:
   - Download live OHLCV data from Yahoo Finance
   - Engineer all 55+ features
   - Run the 3-stage feature selection and training pipeline
   - Train and tune the Ensemble RF + HGB model
   - Run the expanding-window backtest with full NSE costs
   - Export results to CSV

> **Internet required** for Sections 1–3 (live data via `yfinance`). All other sections run offline.

---

## Outputs

| File | Contents |
|---|---|
| `MAHABANK_V9_backtest_daily.csv` | Date-indexed: `Signal`, `Price`, `Target_Return_Fwd`, `Model_Capital`, `BH_Capital`, `Random_Capital`, `Correct_Prediction`, `Model_Type`, `Features_Used` |
| `MAHABANK_V9_backtest_metrics.csv` | Summary: Total Return %, CAGR %, Sharpe Ratio, Max Drawdown %, Calmar Ratio — for all three strategies |

---

## Disclaimer

> This project is for **educational and research purposes only.** It is not financial advice. Past backtest performance does not guarantee future returns. Stock markets are inherently unpredictable. Do not trade real capital based solely on the outputs of this model.
