import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import joblib
import warnings

print(f"sklearn version: {sklearn.__version__}")
warnings.filterwarnings("ignore", category=UserWarning)

# Configuration
CUTOFF_DATE = pd.Timestamp("2000-01-01", tz="UTC")        # use data from 2000 onward
TRAIN_END_DATE = pd.Timestamp("2021-12-31")               # Restrict training up to end of 2021
HISTORY_LAGS_M = 150                                       # window lengths
HISTORY_LAGS_W = 150
HISTORY_LAGS_D = 150
MODEL_OUT = "mlpmodel.joblib"                             # artifacts bundle
MODEL_DIR = Path("artifacts")
MODEL_DIR.mkdir(exist_ok=True)
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = Path(__file__).resolve().parent / "nasdaq_exchange_daily_price_data_close.csv"

# Helpers
# -----------------------
def find_first_datetime_row(csv_path: str) -> int:
    """Find first row whose first column parses as datetime (to skip junk header rows)."""
    raw = pd.read_csv(csv_path, header=None, nrows=500, low_memory=False)
    for idx, val in enumerate(raw.iloc[:, 0]):
        try:
            pd.to_datetime(val)
            return idx
        except Exception:
            continue
    return 0

def load_wide_prices(csv_path: str) -> pd.DataFrame:
    """Load wide daily prices (date + tickers), clean, cut from CUTOFF_DATE."""
    first_data_row = find_first_datetime_row(csv_path)
    df = pd.read_csv(csv_path, skiprows=range(first_data_row), low_memory=False)
    df.rename(columns={df.columns[0]: "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[df["date"] >= CUTOFF_DATE].reset_index(drop=True)
    return df

def wide_to_prices(df_wide: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    """Convert wide daily stock prices into long-format prices with specified frequency."""
    long_df = pd.melt(df_wide, id_vars=["date"], var_name="stock", value_name="price")
    long_df["price"] = pd.to_numeric(long_df["price"], errors="coerce")
    long_df = long_df.dropna(subset=["price"])
    long_df = long_df[long_df["price"] > 0]
    long_df["date"] = pd.to_datetime(long_df["date"])

    def resample_prices(g, freq):
        g = g.sort_values("date").set_index("date")
        if freq == "D":
            out = g[["price"]].reset_index()
        else:
            out = g["price"].resample(freq).last().dropna().reset_index()
        out["stock"] = g["stock"].iloc[0]
        return out

    resampled_list = [
        resample_prices(g, freq) for _, g in long_df.groupby("stock", sort=False)
    ]
    final = pd.concat(resampled_list, ignore_index=True)
    final = final.dropna(subset=["price"]).sort_values(["stock", "date"]).reset_index(drop=True)
    return final

def build_supervised_multiscale(
    monthly: pd.DataFrame,
    weekly: pd.DataFrame,
    daily: pd.DataFrame,
    lags_m: int = 100,
    lags_w: int = 100,
    lags_d: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        try:
            if getattr(out["date"].dt.tz, "tz", None) is not None:
                out["date"] = out["date"].dt.tz_localize(None)
        except Exception:
            pass
        out = out.dropna(subset=["date", "price"])
        out = out.sort_values(["stock", "date"]).reset_index(drop=True)
        out["logret"] = out.groupby("stock", sort=False)["price"] \
                           .transform(lambda p: np.log(p).diff())
        return out

    m = _prep(monthly)
    w = _prep(weekly)
    dly = _prep(daily)

    stocks = sorted(set(m["stock"]))
    X_list, Y_list, D_list, Stock_list = [], [], [], []

    m_groups = {s: g.reset_index(drop=True) for s, g in m.groupby("stock", sort=False)}
    w_groups = {s: g.reset_index(drop=True) for s, g in w.groupby("stock", sort=False)}
    d_groups = {s: g.reset_index(drop=True) for s, g in dly.groupby("stock", sort=False)}

    for s in stocks:
        if s not in w_groups or s not in d_groups:
            continue

        gm = m_groups[s]
        gw = w_groups[s]
        gd = d_groups[s]

        m_dates = gm["date"].to_numpy()
        m_ret = gm["logret"].to_numpy()
        w_dates = gw["date"].to_numpy()
        w_ret = gw["logret"].to_numpy()
        d_dates = gd["date"].to_numpy()
        d_ret = gd["logret"].to_numpy()

        for t in range(1, len(gm) - 1):
            t_date = m_dates[t]

            if t < lags_m:
                continue
            m_window = m_ret[t - lags_m + 1 : t + 1]
            if np.any(np.isnan(m_window)) or len(m_window) != lags_m:
                continue

            idx_w_end = np.searchsorted(w_dates, t_date, side="right") - 1
            if idx_w_end < 0 or idx_w_end + 1 < lags_w:
                continue
            w_window = w_ret[idx_w_end - lags_w + 1 : idx_w_end + 1]
            if np.any(np.isnan(w_window)) or len(w_window) != lags_w:
                continue

            idx_d_end = np.searchsorted(d_dates, t_date, side="right") - 1
            if idx_d_end < 0 or idx_d_end + 1 < lags_d:
                continue
            d_window = d_ret[idx_d_end - lags_d + 1 : idx_d_end + 1]
            if np.any(np.isnan(d_window)) or len(d_window) != lags_d:
                continue

            label = m_ret[t+1]
            if np.isnan(label):
                continue

            x = np.concatenate([m_window, w_window, d_window], dtype=np.float32)
            X_list.append(x.astype(np.float32))
            Y_list.append(label)
            D_list.append(t_date)
            Stock_list.append(s)

    if not X_list:
        raise ValueError("No samples built. Check histories and features.")

    X = np.vstack(X_list).astype(np.float32)
    y = np.asarray(Y_list, dtype=np.float32)
    d_out = np.asarray(D_list)
    stocks = np.asarray(Stock_list, dtype=object)
    return X, y, d_out, stocks

def get_train_mask(dates: np.ndarray):
    """Create a training mask encompassing records up to the end of 2021."""
    dates = pd.to_datetime(dates)
    if getattr(dates, "tz", None) is not None:
        dates = dates.tz_localize(None)

    train_mask = dates <= TRAIN_END_DATE
    print(f"Dataset Split: Training on data up to {TRAIN_END_DATE.date()} ({train_mask.sum()} samples)")
    return train_mask

CACHE_DIR = Path("cache_simple")
CACHE_DIR.mkdir(exist_ok=True)

def cache_load(name):
    path = CACHE_DIR / f"{name}.joblib"
    if path.exists():
        print(f"🔄 Loading cached {name} from {path}")
        return joblib.load(path)
    return None

def cache_save(name, obj):
    path = CACHE_DIR / f"{name}.joblib"
    joblib.dump(obj, path)
    print(f"💾 Saved {name} to cache → {path}")

def save_model_bundle(name, bundle):
    path = MODEL_DIR / f"{name}.joblib"
    joblib.dump(bundle, path)
    print(f"✅ Saved model artifact → {path}")

def main():
    t0 = time.time()
    print("Loading daily wide prices…")

    wide = cache_load("wide_exact")
    if wide is None:
        print("Loading daily wide prices… (first time)")
        wide = load_wide_prices(DATA_FILE)
        cache_save("wide_exact", wide)
    else:
        print("Using cached wide data")
        
    print(f"✅ Rows after cutoff {CUTOFF_DATE.date()}: {len(wide):,}")
    print(f"📅 Date range: {wide['date'].min()} → {wide['date'].max()}")
    
    print("Converting to monthly, daily and weekly prices…")
    monthly_prices = wide_to_prices(wide, freq="ME")
    weekly_prices  = wide_to_prices(wide, freq="W")
    daily_prices   = wide_to_prices(wide, freq="D")

    print("Building supervised dataset (rolling windows)…")
    supervised = None

    if supervised is None:
        print("Building supervised dataset (first time) …")
        X, y, d, stocks = build_supervised_multiscale(
            monthly=monthly_prices,
            weekly=weekly_prices,
            daily=daily_prices,
            lags_m=HISTORY_LAGS_M,
            lags_w=HISTORY_LAGS_W,
            lags_d=HISTORY_LAGS_D,
        )
        cache_save("supervised_arrays_exact_2", (X, y, d, stocks))
        joblib.dump(stocks, "artifacts/stocks.joblib")
    else:
        print("Using cached supervised arrays (X, y, d, stocks)")
        X, y, d, stocks = supervised

    print(f"✅ Samples: {X.shape[0]:,} | Features: {X.shape[1]}")

    # Generate training mask restricted to <= 2021-12-31
    train_mask = get_train_mask(d)
    
    if train_mask.sum() == 0:
        raise ValueError("No training samples found before the 2021-12-31 cutoff. Check your source data dates.")

    # --- Split features by scale ---
    lags_m, lags_w, lags_d = HISTORY_LAGS_M, HISTORY_LAGS_W, HISTORY_LAGS_D
    X_m = X[:, :lags_m]
    X_w = X[:, lags_m:lags_m+lags_w]
    X_d = X[:, lags_m+lags_w:]

    # Helper to train model parameters on filtered matrix
    def train_model(X_data, label):
        X_train = X_data[train_mask]
        y_train = y[train_mask]

        # Scale features using only the selected training subset
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        # Define model setup
        model = MLPRegressor(
            hidden_layer_sizes=(32,16, 8),
            activation="relu",
            solver="adam",
            learning_rate_init=0.005,
            learning_rate="adaptive",
            alpha=0.001,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.10,  # Internal validation used solely for MLP early stopping
            random_state=42,
            verbose=True,
        )

        print(f"\n🧠 Training {label} model on filtered dataset…")
        model.fit(X_train, y_train)
        return {
            "model": model,
            "scaler": scaler,
        }

    monthly_out = train_model(X_m, "Monthly")
    weekly_out  = train_model(X_w, "Weekly")
    daily_out   = train_model(X_d, "Daily")

    elapsed = time.time() - t0
    last_train_date = pd.to_datetime(d[train_mask]).max()
    print(f"\n⏱️ Total runtime: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")
    
    print("\n💾 Saving models and scalers...")
    save_model_bundle(
        "monthly_reg",
        {
            "model": monthly_out["model"],
            "scaler": monthly_out["scaler"],
            "lags": HISTORY_LAGS_M,
            "frequency": "monthly",
            "trained_until": last_train_date,
            "sklearn_version": sklearn.__version__,
        },
    )

    save_model_bundle(
        "weekly_reg_good",
        {
            "model": weekly_out["model"],
            "scaler": weekly_out["scaler"],
            "lags": HISTORY_LAGS_W,
            "frequency": "weekly",
            "trained_until": last_train_date,
            "sklearn_version": sklearn.__version__,
        },
    )

    save_model_bundle(
        "daily_reg",
        {
            "model": daily_out["model"],
            "scaler": daily_out["scaler"],
            "lags": HISTORY_LAGS_D,
            "frequency": "daily",
            "trained_until": last_train_date,
            "sklearn_version": sklearn.__version__,
        },
    )
    print("✅ All models and scalers saved successfully using data up to 2021!")

if __name__ == "__main__":
    main()