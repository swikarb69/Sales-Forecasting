<div align="center">

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Space+Grotesk&weight=700&size=32&duration=3000&pause=1000&color=A78BFA&center=true&vCenter=true&width=800&height=70&lines=Sales+Forecasting+Pipeline;XGBoost+%7C+3M%2B+Rows+%7C+Time+Series)](https://git.io/typing-svg)

> 🤖 **Note:** Encountered a pandas indexing error during development (`IndexError: Too many levels`)
> on the rolling statistics computation. Debugged and resolved with AI assistance —
> `reset_index(level=[0,1])` was replaced with `groupby.transform(lambda x: ...)` to correctly
> preserve index alignment across grouped rolling windows.

<img src="https://capsule-render.vercel.app/api?type=rect&height=3&color=gradient&customColorList=12" width="100%"/>

![Python](https://img.shields.io/badge/Python-A78BFA?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-818CF8?style=for-the-badge&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-C084FC?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-7C3AED?style=for-the-badge&logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-818CF8?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Shipped-A78BFA?style=for-the-badge)

> **Predicting daily sales across 54 stores and 33 product families in Ecuador.**
> Trained on 3M+ rows of retail transaction data with lag features, rolling statistics, holiday signals, and oil price as an economic indicator.

</div>

<img src="https://capsule-render.vercel.app/api?type=rect&height=2&color=gradient&customColorList=12" width="100%"/>

---

## `◈` The Problem

A retail chain operating across Ecuador wants to know: **"How much of each product will each store sell on a given day?"**

Getting this right has real consequences:
- **Understock** → lost sales, unhappy customers
- **Overstock** → waste, spoilage, tied-up capital

The challenge: sales are driven by complex interacting signals — day of week, holidays, oil price (Ecuador's economy is oil-dependent), promotions, and each store's own historical patterns.

---

## `◈` Approach

```python
pipeline = {
    "dataset"    : "Kaggle Store Sales — Time Series Forecasting (3M+ rows)",
    "model"      : "XGBoostRegressor",
    "features"   : ["lag features", "rolling stats", "seasonality", "oil price", "holidays"],
    "evaluation" : "RMSLE (Root Mean Squared Log Error)",
    "split"      : "Last 28 days held out as validation",
}
```

**Why XGBoost over a pure time series model (ARIMA/Prophet)?**
With 54 stores × 33 product families = 1,782 separate series, fitting individual time series models is impractical. XGBoost learns shared patterns across all series simultaneously — and with the right lag/rolling features, it captures temporal dynamics just as well.

---

## `◈` Results

<div align="center">

| Metric | Value |
|:---|:---:|
| **RMSLE (validation)** | **0.6035** |
| **Train rows** | 1,905,552 |
| **Validation rows** | 35,640 |
| **Top feature** | `rolling_mean_7` (61.7% importance) |
| **Model** | XGBoostRegressor · 500 estimators |

</div>

The rolling 7-day mean dominates feature importance at **61.7%** — confirming that recent sales history is the strongest signal. Rolling 14-day mean adds another **24%**, while lag features and external signals (promotions, holidays, oil price) fill in the rest.

---

## `◈` Feature Engineering

The model sees no raw time series — only engineered features:

**Lag Features** — what sales looked like in the past
```python
lag_7   → sales 1 week ago (same weekday last week)
lag_14  → sales 2 weeks ago
lag_28  → sales 4 weeks ago (same weekday last month)
```

**Rolling Statistics** — smoothed historical signal
```python
rolling_mean_7   → average sales over last 7 days
rolling_mean_14  → average sales over last 14 days
rolling_std_7    → volatility over last 7 days
rolling_max_14   → peak sales over last 14 days
```

**Seasonality Features** — calendar signals
```python
day_of_week, week, month, quarter, year
is_weekend      → binary flag
```

**External Signals**
```python
oil_price       → Ecuador's economy is oil-dependent; forward-filled
is_holiday      → national holidays from holidays_events.csv
onpromotion     → number of items on promotion (provided in dataset)
transactions    → daily footfall per store
```

> ⚠️ All lag/rolling features are computed **after** the train/val split to prevent data leakage.

---

## `◈` Pipeline

```
raw CSVs (train, stores, oil, holidays, transactions)
   │
   ▼
① Merge all supporting tables onto the main train dataframe
   │
   ▼
② Feature Engineering
   ├── Date decomposition (year, month, week, day_of_week, is_weekend)
   ├── Lag features per store-family group (lag_7, lag_14, lag_28)
   ├── Rolling statistics (mean, std, max)
   └── Label encode categoricals (family, city, state, store type)
   │
   ▼
③ Train / Validation split
   └── Last 28 days held out → ~1,782 × 28 rows for validation
   │
   ▼
④ Train XGBoostRegressor
   └── 500 estimators · depth 6 · lr 0.05 · subsample 0.8
   │
   ▼
⑤ Evaluate on validation set
   └── RMSLE · Feature importance · Sample predictions
```

---

## `◈` Dataset

**[Store Sales — Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)** (Kaggle)

| File | Description |
|:---|:---|
| `train.csv` | 3M+ rows — date, store, family, sales, onpromotion |
| `stores.csv` | Store metadata — city, state, type, cluster |
| `oil.csv` | Daily oil prices (Ecuador is oil-dependent) |
| `holidays_events.csv` | National/regional/local holidays |
| `transactions.csv` | Daily transactions per store |

Place all files in `data/` before running.

---

## `◈` Tech Stack

<div align="center">

| Tool | Purpose |
|:---|:---|
| `Python` | Core language |
| `XGBoost` | Gradient boosted trees — main model |
| `Pandas` | Data loading, merging, feature engineering |
| `NumPy` | Numerical operations, clipping predictions |
| `Scikit-Learn` | Label encoding, RMSLE metric |

</div>

---

## `◈` Run It

```bash
# Clone the repo
git clone https://github.com/swikarb69/Sales-Forecasting.git
cd Sales-Forecasting

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle and place CSVs in data/
# then run:
python main.py
```

**Expected output:**
```
Loading data...
  Train shape : (3000888, 6)
Merging supporting data...
Engineering features...
Splitting train / validation...
  Train rows : 1,905,552
  Val rows   : 35,640
Training XGBoost model...
[0]    validation_0-rmse: 1110.26943
[100]  validation_0-rmse:  187.66082
[200]  validation_0-rmse:  181.21753
[300]  validation_0-rmse:  178.91842
[400]  validation_0-rmse:  177.30050
[499]  validation_0-rmse:  176.43784

  RMSLE (validation) : 0.6035

  Top 10 Feature Importances:
        Feature  Importance
 rolling_mean_7    0.6169
rolling_mean_14    0.2413
          lag_7    0.0427
         lag_14    0.0161
    onpromotion    0.0096
         lag_28    0.0092
   transactions    0.0076
     is_holiday    0.0069
    day_of_week    0.0062
            day    0.0055
Done! Model trained and evaluated successfully.
```

---

## `◈` Key Design Decisions

**1. Lag features as temporal memory**
XGBoost has no built-in notion of time. Lag features are how we give the model memory — `lag_7` captures the same-weekday-last-week pattern which is the strongest signal in retail data.

**2. Rolling on shifted data only**
`rolling_mean_7` is computed on `shift(1)` — meaning it looks at days *before* the prediction day. Using the current day's sales in a rolling window would be leakage.

**3. Oil price as an economic macro signal**
Ecuador's economy is strongly correlated with oil prices. Including it as a feature lets the model capture macro-economic effects on consumer spending — something pure lag features miss.

**4. RMSLE over RMSE**
Log-error penalises under-prediction more than over-prediction and handles the wide range of sales magnitudes (some products sell 0–10 units/day, others sell thousands) without letting high-volume items dominate the loss.

**5. `tree_method="hist"`**
With 3M+ rows, standard XGBoost is slow. The histogram-based method bins continuous features first — dramatically faster on large datasets with no meaningful accuracy loss.

---

## `◈` Next Steps

- [ ] **LightGBM** — often faster and slightly more accurate on tabular data at scale
- [ ] **Per-family models** — some product families (Produce vs Electronics) have very different seasonality patterns
- [ ] **SHAP values** — explain which features drive predictions for a specific store/day
- [ ] **Streamlit dashboard** — visualise forecasts vs actuals interactively
- [ ] **Recursive forecasting** — use predicted values as lag inputs to forecast further into the future

---

## `◈` Project Structure

```
Sales-Forecasting/
│
├── data/
│   ├── train.csv
│   ├── stores.csv
│   ├── oil.csv
│   ├── holidays_events.csv
│   └── transactions.csv
│
├── main.py              # Full ML pipeline
├── requirements.txt     # Dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=rect&height=2&color=gradient&customColorList=12" width="100%"/>

**Built by [Swikar Bhattarai](https://github.com/swikarb69) · Nepal 🇳🇵**

[![GitHub](https://img.shields.io/badge/GitHub-swikarb69-A78BFA?style=for-the-badge&logo=github&logoColor=white)](https://github.com/swikarb69)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Swikar_Bhattarai-818CF8?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/swikar-bhattarai-11178b240)

*"The universe speaks in patterns. I speak Python."*

</div>
