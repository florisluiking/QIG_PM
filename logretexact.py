
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
import sklearn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, f1_score, precision_score
from sklearn.neural_network import MLPRegressor
import joblib
import warnings
import json
print(f"sklearn version: {sklearn.__version__}")

warnings.filterwarnings("ignore", category=UserWarning)

CUTOFF_DATE = pd.Timestamp("2000-01-01", tz="UTC")        # use data from 2000 onward
HISTORY_LAGS_M = 150                                        # window length
HISTORY_LAGS_W = 150
HISTORY_LAGS_D = 100
MODEL_OUT = "mlpmodel.joblib"               # artifacts bundle
VAL_MONTHS = 0   # last 72 months of training used for validation
TEST_MONTHS = 36  # last 72 months used for test
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
    # first column is date, possibly called something else
    df.rename(columns={df.columns[0]: "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"])
    # cutoff
    df = df[df["date"] >= CUTOFF_DATE].reset_index(drop=True)
    return df

def wide_to_prices(df_wide: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    """
    Convert wide daily stock prices into long-format prices
    with user-specified frequency: daily, weekly, or monthly.

    Parameters
    ----------
    df_wide : pd.DataFrame
        Wide-format DataFrame with columns: ['date', stock1, stock2, ...]
    freq : str, optional
        Resampling frequency. Options:
        - 'D' for daily (no resampling)
        - 'W' for weekly (end-of-week)
        - 'M' for monthly (end-of-month)
        Default is 'M'.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns ['date', 'stock', 'price'].
    """

    # Melt wide data to long format
    long_df = pd.melt(df_wide, id_vars=["date"], var_name="stock", value_name="price")
    long_df["price"] = pd.to_numeric(long_df["price"], errors="coerce")
    long_df = long_df.dropna(subset=["price"])
    long_df = long_df[long_df["price"] > 0]



    # Ensure datetime format
    long_df["date"] = pd.to_datetime(long_df["date"])

    # Define resampling helper
    def resample_prices(g, freq):
        g = g.sort_values("date").set_index("date")

        if freq == "D":
            # Keep daily data (no resampling)
            out = g[["price"]].reset_index()
        else:
            # Use pandas resample for weekly ('W') or monthly ('M')
            out = g["price"].resample(freq).last().dropna().reset_index()

        out["stock"] = g["stock"].iloc[0]
        return out

    # Apply for each stock
    resampled_list = [
        resample_prices(g, freq) for _, g in long_df.groupby("stock", sort=False)
    ]

    # Combine and clean up
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def _prep(df: pd.DataFrame) -> pd.DataFrame: #filters nan, makes log returns and formats dates correctly
        out = df.copy()
        # normalize tz: make timezone-naive for safe alignment/comparison
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        try:
            if getattr(out["date"].dt.tz, "tz", None) is not None:
                out["date"] = out["date"].dt.tz_localize(None)
        except Exception:
            # pandas accessor may behave differently across versions; fallback noop
            pass
        out = out.dropna(subset=["date", "price"])
        out = out.sort_values(["stock", "date"]).reset_index(drop=True)
        # log-returns aligned to the *period end date* (same index as price; first is NaN)
        out["logret"] = out.groupby("stock", sort=False)["price"] \
                           .transform(lambda p: np.log(p).diff())
        return out

    m = _prep(monthly)
    w = _prep(weekly)
    dly = _prep(daily)

    # Build quick per-stock views
    stocks = sorted(set(m["stock"]))  # we’ll skip a stock if it lacks weekly/daily windows
    X_list, Y_list, D_list, Stock_list = [], [], [], []

    # Grouped frames for faster lookup
    m_groups = {s: g.reset_index(drop=True) for s, g in m.groupby("stock", sort=False)}
    w_groups = {s: g.reset_index(drop=True) for s, g in w.groupby("stock", sort=False)}
    d_groups = {s: g.reset_index(drop=True) for s, g in dly.groupby("stock", sort=False)}

    for s in stocks:
        if s not in w_groups or s not in d_groups:
            continue  # need all three frequencies for this stock

        gm = m_groups[s]
        gw = w_groups[s]
        gd = d_groups[s]

        m_dates = gm["date"].to_numpy()
        m_prices = gm["price"].to_numpy()
        m_ret = gm["logret"].to_numpy()   # aligned to m_dates (first NaN)

        w_dates = gw["date"].to_numpy()
        w_ret = gw["logret"].to_numpy()

        d_dates = gd["date"].to_numpy()
        d_ret = gd["logret"].to_numpy()

        # We can form a label at monthly index t if t+1 exists
        # And we need full windows ending at monthly date m_dates[t]
        for t in range(1, len(gm) - 1):  # start at 1 because first monthly return is NaN
            t_date = m_dates[t]

            # --- Monthly window (exact index t)
            if t < lags_m:
                continue
            m_window = m_ret[t - lags_m + 1 : t + 1]  # length lags_m; ends at t
            if np.any(np.isnan(m_window)) or len(m_window) != lags_m:
                continue

            # --- Weekly window: take last lags_w weekly returns with date <= t_date
            # Find the rightmost weekly index with date <= t_date
            idx_w_end = np.searchsorted(w_dates, t_date, side="right") - 1
            if idx_w_end < 0 or idx_w_end + 1 < lags_w:
                continue
            w_window = w_ret[idx_w_end - lags_w + 1 : idx_w_end + 1]
            if np.any(np.isnan(w_window)) or len(w_window) != lags_w:
                continue

            # --- Daily window: last lags_d daily returns with date <= t_date
            idx_d_end = np.searchsorted(d_dates, t_date, side="right") - 1
            if idx_d_end < 0 or idx_d_end + 1 < lags_d:
                continue
            d_window = d_ret[idx_d_end - lags_d + 1 : idx_d_end + 1]
            if np.any(np.isnan(d_window)) or len(d_window) != lags_d:
                continue
            #predicted value based on output
            label = m_ret[t+1]
            if np.isnan(label):
                continue
            # Concatenate features in fixed order: [monthly | weekly | daily]
            x = np.concatenate([m_window, w_window, d_window], dtype=np.float32)
            X_list.append(x.astype(np.float32))
            Y_list.append(label)
            D_list.append(t_date)
            Stock_list.append(s)

    if not X_list:
        raise ValueError("No samples built. Check lags and that all three frequencies have sufficient history.")

    X = np.vstack(X_list).astype(np.float32)
    y_raw = np.asarray(Y_list, dtype=np.float32)   # raw log returns
    y = (y_raw > 0).astype(np.float32)              # binary labels
    R = y_raw                                        # realized returns for portfolio sim
    d_out = np.asarray(D_list)
    stocks = np.asarray(Stock_list, dtype=object)
    return X, y,d_out, R, stocks

#-----------------#

def split_masks(dates: np.ndarray, val_months: int = VAL_MONTHS, test_months: int = TEST_MONTHS):
    """
    Dynamically create train, validation, and test masks based on the last date in the dataset.
    """
    dates = pd.to_datetime(dates)
    if getattr(dates, "tz", None) is not None:
        dates = dates.tz_localize(None)

    last_date = dates.max()  # last available date in dataset

    # Compute split cutoffs
    train_end  = last_date - pd.DateOffset(months=(val_months + test_months))
    val_start  = train_end + pd.DateOffset(months=1)
    val_end    = last_date - pd.DateOffset(months=test_months)
    test_start = val_end + pd.DateOffset(months=1)
    test_end   = last_date

    # Boolean masks
    train_core_mask = dates <= train_end
    val_mask        = (dates >= val_start) & (dates <= val_end)
    test_mask       = (dates >= test_start) & (dates <= test_end)

    print("Dynamic split dates:")
    print(f"  Train core:   up to {train_end.date()} ({train_core_mask.sum()} samples)")
    print(f"  Validation:   {val_start.date()} → {val_end.date()} ({val_mask.sum()} samples)")
    print(f"  Test:         {test_start.date()} → {test_end.date()} ({test_mask.sum()} samples)")

    return train_core_mask, val_mask, test_mask

CACHE_DIR = Path("cache_simple")
CACHE_DIR.mkdir(exist_ok=True)


def cache_load(name):
    """Load cached object if exists, else return None."""
    path = CACHE_DIR / f"{name}.joblib"
    if path.exists():
        print(f"🔄 Loading cached {name} from {path}")
        return joblib.load(path)
    return None

def cache_save(name, obj):
    """Save object to cache."""
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
    if monthly_prices is None:
        print("Converting daily → monthly/weekly/daily (first time)")
        monthly_prices = wide_to_prices(wide, freq="ME")
        weekly_prices  = wide_to_prices(wide, freq="W")
        daily_prices   = wide_to_prices(wide, freq="D")

        cache_save("monthly_prices_exact", monthly_prices)
        cache_save("weekly_prices_exact", weekly_prices)
        cache_save("daily_prices_exact",  daily_prices)
    else:
        print("Using cached monthly/weekly/daily")
    print(f"✅ Monthly rows: {len(monthly_prices):,}  | stocks: {monthly_prices['stock'].nunique():,}")
    print(f"📅 Monthly date range: {monthly_prices['date'].min()} → {monthly_prices['date'].max()}")

    print("Building supervised dataset (rolling windows)…")
    supervised = cache_load("supervised_arrays_exact_2")
    supervised = None
    if supervised is None:
        print("Building supervised dataset (first time) …")
        X, y, d, R, stocks = build_supervised_multiscale(
            monthly=monthly_prices,
            weekly=weekly_prices,
            daily=daily_prices,
            lags_m=HISTORY_LAGS_M,
            lags_w=HISTORY_LAGS_W,
            lags_d=HISTORY_LAGS_D,
        )
        cache_save("supervised_arrays_new4", (X, y, d, R, stocks))
        joblib.dump(stocks, "artifacts/stocks.joblib")
    else:
        print("Using cached supervised arrays (X, y, d, stocks)")
        X, y, d, stocks = supervised

    print(f"✅ Samples: {X.shape[0]:,} | Features: {X.shape[1]} "
          f"(M:{HISTORY_LAGS_M} + W:{HISTORY_LAGS_W} + D:{HISTORY_LAGS_D}) "
          f"| Positives: {y.sum():,} ({y.mean()*100:.2f}%)")

    # Split into train/val/test
    train_core_mask, val_mask, test_mask = split_masks(d, val_months=VAL_MONTHS, test_months=TEST_MONTHS)

    # --- Split features by scale ---
    lags_m, lags_w, lags_d = HISTORY_LAGS_M, HISTORY_LAGS_W, HISTORY_LAGS_D
    X_m = X[:, :lags_m]
    X_w = X[:, lags_m:lags_m+lags_w]
    X_d = X[:, lags_m+lags_w:]

    # Helper to train and evaluate one model
    def train_and_get_probs(X_data, label):
        # Split data
        X_train, X_val, X_test = X_data[train_core_mask], X_data[val_mask], X_data[test_mask]
        y_train = y[train_core_mask]

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)
        X_test  = scaler.transform(X_test)
        # Define model
        model = MLPRegressor(
            hidden_layer_sizes=(8,4),
            activation="relu",
            solver="adam",
            learning_rate_init=0.005,
            learning_rate="adaptive",
            alpha=0.001,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.10,
            random_state=42,
            verbose=True,
            )

        print(f"\n🧠 Training {label} model…")
        model.fit(X_train, y_train)
        y_pred_val = model.predict(X_val)
        return {
            "model": model,
            "scaler": scaler,
            "y_pred_val": y_pred_val,
            "y_val": y_val,
        }

    y_val = y[val_mask] 
    y_val_binary = (y_val > 0).astype(int)
    monthly_out = train_and_get_probs(X_m, "Monthly")
    weekly_out  = train_and_get_probs(X_w, "Weekly")
    daily_out   = train_and_get_probs(X_d, "Daily")
    y_pred_m = monthly_out["y_pred_val"]
    y_pred_w = weekly_out["y_pred_val"]
    y_pred_d = daily_out["y_pred_val"]

    # --- Test multiple thresholds efficiently ---
    thresholds = np.arange(-0.02, 0.02, 0.002)

    results = []

    for thr in thresholds:
        # Consensus signal
        signal_mask = (
            (y_pred_m >= thr) &
            (y_pred_w >= thr) &
            (y_pred_d >= thr)
        )

        n_signals = signal_mask.sum()
        if n_signals == 0:
            continue

        realized_returns = y_val[signal_mask]

        results.append({
            "threshold": thr,
            "coverage": n_signals / len(y_val),
            "n_signals": n_signals,
            "hit_rate": np.mean(realized_returns > 0),
            "mean_return": np.mean(realized_returns),
            "median_return": np.median(realized_returns),
            "variance": np.var(realized_returns),
            "min_return": np.min(realized_returns),
            "max_return": np.max(realized_returns),
            "quantile_5": np.percentile(realized_returns, 5),
            "quantile_10": np.percentile(realized_returns, 10),
            "quantile_90": np.percentile(realized_returns, 90),
            "quantile_95": np.percentile(realized_returns, 95),
        })
    # ----------------------------------------------------
    # Results table
    # ----------------------------------------------------
    df_results = pd.DataFrame(results).sort_values("threshold").reset_index(drop=True)
    df_results.to_csv("df_results.csv", index=False)

    print("\n=== Threshold evaluation results ===")
    print(df_results)

    # Focus on economically meaningful region
    print("\n=== Signals with at least 50 observations ===")
    print(df_results[df_results["n_signals"] >= 50])

    # ----------------------------------------------------
    # Plot: Coverage vs Mean Return (Efficient Frontier)
    # ----------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(df_results["coverage"], df_results["mean_return"], marker="o")
    plt.xlabel("Coverage (fraction of months with signal)")
    plt.ylabel("Mean realized next-month return")
    plt.title("Signal quality vs frequency (validation set)")
    plt.grid(True)
    plt.show()
    thresholds = np.arange(-0.1, 0.1,0.02)  # you can change this range
    results = []


    elapsed = time.time() - t0
    
    last_train_date = pd.to_datetime(d[train_core_mask]).max()
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
        "weekly_reg",
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
    print("✅ All models, scalers, and thresholds saved successfully!")

if __name__ == "__main__":
    main()
