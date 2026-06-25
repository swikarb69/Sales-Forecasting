import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
print("Loading data...")

train        = pd.read_csv("data/train.csv", parse_dates=["date"])
stores       = pd.read_csv("data/stores.csv")
oil          = pd.read_csv("data/oil.csv", parse_dates=["date"])
holidays     = pd.read_csv("data/holidays_events.csv", parse_dates=["date"])
transactions = pd.read_csv("data/transactions.csv", parse_dates=["date"])

print(f"  Train shape : {train.shape}")

# ─────────────────────────────────────────
# 2. MERGE SUPPORTING TABLES
# ─────────────────────────────────────────
print("Merging supporting data...")

# Stores metadata
train = train.merge(stores, on="store_nbr", how="left")

# Oil price (forward-fill missing values)
oil = oil.rename(columns={"dcoilwtico": "oil_price"})
oil["oil_price"] = oil["oil_price"].ffill()
train = train.merge(oil, on="date", how="left")

# Transactions
train = train.merge(transactions, on=["date", "store_nbr"], how="left")

# Holidays — keep only national holidays to avoid complexity
national_holidays = holidays[
    (holidays["locale"] == "National") & (holidays["transferred"] == False)
][["date", "type"]].rename(columns={"type": "holiday_type"})
train = train.merge(national_holidays, on="date", how="left")
train["is_holiday"] = train["holiday_type"].notna().astype(int)

# ─────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────
print("Engineering features...")

# Sort for lag/rolling calculations
train = train.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

# Date features
train["year"]        = train["date"].dt.year
train["month"]       = train["date"].dt.month
train["day"]         = train["date"].dt.day
train["day_of_week"] = train["date"].dt.dayofweek
train["week"]        = train["date"].dt.isocalendar().week.astype(int)
train["is_weekend"]  = (train["day_of_week"] >= 5).astype(int)
train["quarter"]     = train["date"].dt.quarter

# Lag features — per store & family group
grp = train.groupby(["store_nbr", "family"])["sales"]

train["lag_7"]  = grp.shift(7)
train["lag_14"] = grp.shift(14)
train["lag_28"] = grp.shift(28)

# Rolling statistics
train["rolling_mean_7"]  = grp.transform(lambda x: x.shift(1).rolling(7).mean())
train["rolling_mean_14"] = grp.transform(lambda x: x.shift(1).rolling(14).mean())
train["rolling_std_7"]   = grp.transform(lambda x: x.shift(1).rolling(7).std())
train["rolling_max_14"]  = grp.transform(lambda x: x.shift(1).rolling(14).max())

# Encode categoricals
le = LabelEncoder()
train["family_enc"] = le.fit_transform(train["family"])
train["city_enc"]   = le.fit_transform(train["city"])
train["state_enc"]  = le.fit_transform(train["state"])
train["type_enc"]   = le.fit_transform(train["type"])

# ─────────────────────────────────────────
# 4. TRAIN / VALIDATION SPLIT
# ─────────────────────────────────────────
print("Splitting train / validation...")

# Use last 28 days as validation
cutoff = train["date"].max() - pd.Timedelta(days=28)

df_train = train[train["date"] <= cutoff].copy()
df_val   = train[train["date"] >  cutoff].copy()

FEATURES = [
    "store_nbr", "family_enc", "city_enc", "state_enc", "type_enc",
    "cluster", "oil_price", "transactions", "is_holiday",
    "year", "month", "day", "day_of_week", "week", "is_weekend", "quarter",
    "onpromotion",
    "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_14", "rolling_std_7", "rolling_max_14",
]
TARGET = "sales"

# Drop rows with NaN from lag creation
df_train = df_train.dropna(subset=FEATURES)
df_val   = df_val.dropna(subset=FEATURES)

X_train = df_train[FEATURES]
y_train = df_train[TARGET]
X_val   = df_val[FEATURES]
y_val   = df_val[TARGET]

print(f"  Train rows : {len(X_train):,}")
print(f"  Val rows   : {len(X_val):,}")

# ─────────────────────────────────────────
# 5. MODEL TRAINING
# ─────────────────────────────────────────
print("Training XGBoost model...")

model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",   # fast on large data
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100,
)

# ─────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────
print("\nEvaluating...")

y_pred = model.predict(X_val)
y_pred = np.clip(y_pred, 0, None)   # sales can't be negative

# RMSLE — standard metric for this competition
rmsle = np.sqrt(mean_squared_log_error(
    np.clip(y_val, 0, None),
    y_pred
))
print(f"\n  RMSLE (validation) : {rmsle:.4f}")

# Feature importance
print("\n  Top 10 Feature Importances:")
importance = pd.DataFrame({
    "Feature"   : FEATURES,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

print(importance.head(10).to_string(index=False))

# ─────────────────────────────────────────
# 7. SAMPLE PREDICTIONS
# ─────────────────────────────────────────
print("\n  Sample Predictions vs Actuals:")
sample = df_val[["date", "store_nbr", "family", "sales"]].copy()
sample["predicted"] = y_pred
sample["predicted"] = sample["predicted"].round(2)
print(sample.head(10).to_string(index=False))

print("\nDone! Model trained and evaluated successfully.")