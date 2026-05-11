
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
import sklearn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, f1_score, precision_score, log_loss
import joblib
import warnings
import json

print(f"Using sklearn {sklearn.__version__}")
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"   # macOS Accelerate
os.environ["NUMEXPR_NUM_THREADS"] = "1"

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(42)
# Config
# -----------------------
DATA_FILE = 'nasdaq_exchange_daily_price_data_close.csv'  # wide: date + tickers
CUTOFF_DATE = pd.Timestamp("2000-01-01", tz="UTC")
HISTORY_LAGS_M = 100                                        # default window length
HISTORY_LAGS_W = 100
HISTORY_LAGS_D = 100
MODEL_OUT = "mlp_monthly_calibrated.joblib"               # artifacts bundle
VAL_MONTHS = 36   # last 24 months of training used for validation
TEST_MONTHS = 36  # last 12 months used for test

# Hyperparameter search config for a single frequency
# Set RUN_SINGLE_SCALE_HYPEROPT = True to enable
RUN_SINGLE_SCALE_HYPEROPT = True
HYPEROPT_SCALE = "weekly"  # one of {"monthly", "weekly", "daily"}
HYPEROPT_LAGS = [50,100, 150]  # candidate history lengths for the chosen scale
HYPEROPT_LAYER_CONFIGS = [
    (128,64),
    (32,16,8),
    (8, 4)
]
HYPEROPT_LEARNING_RATES = [0.0001, 0.0005, 0.001, 0.005]

MODEL_DIR = Path("artifacts")
MODEL_DIR.mkdir(exist_ok=True)

print("Made it")

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
    """
    Build (X, y, dates) using aligned rolling windows at three time scales:
    - Monthly:   last `lags_m` monthly log-returns up to month t
    - Weekly:    last `lags_w` weekly  log-returns up to month t
    - Daily:     last `lags_d` daily   log-returns up to month t
    Label: 1 if next month's price > current month's price (from monthly series), else 0.

    Inputs
    ------
    monthly, weekly, daily : long DataFrames with columns ['date','stock','price'].
        Dates should already be EOM/EOW/daily (respectively), sorted asc per stock.
        'date' may be tz-aware or naive; this function handles both consistently.

    Returns
    -------
    X : np.ndarray, shape (n_samples, lags_m + lags_w + lags_d), dtype float32
    y : np.ndarray, shape (n_samples,), dtype int32
    d : np.ndarray, shape (n_samples,), dtype datetime64  (monthly timestamp t used for features)
    """
    def _prep(df: pd.DataFrame) -> pd.DataFrame: # filters nan, makes log returns and formats dates correctly
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
    X_list, Y_list, D_list, Real_list, Stock_list = [], [], [], [], []

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

            # --- Label from monthly prices: direction t -> t+1
            label = int(m_prices[t + 1] > m_prices[t])
            zabel = m_ret[ t +1]
            if np.isnan(zabel):
                continue
            # Concatenate features in fixed order: [monthly | weekly | daily]
            x = np.concatenate([m_window, w_window, d_window], dtype=np.float32)
            X_list.append(x.astype(np.float32))
            Y_list.append(label)
            Real_list.append(zabel)
            D_list.append(t_date)
            Stock_list.append(s)

    if not X_list:
        raise ValueError("No samples built. Check lags and that all three frequencies have sufficient history.")

    X = np.vstack(X_list).astype(np.float32)
    y = np.asarray(Y_list, dtype=np.int32)
    d_out = np.asarray(D_list)
    R = np.asarray(Real_list, dtype=np.float32)
    stocks = np.asarray(Stock_list, dtype=object)
    return X, y, d_out, R, stocks


# -----------------#

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
    path = CACHE_DIR / f"{name}.joblib"
    if path.exists():
        print(f"🔄 Loading cached {name} from {path}")
        return joblib.load(path)
    return None
def cache_save(name, obj):
    """Save object to cache."""
    path = CACHE_DIR / f"{name}.joblib"
    print("SAVE sklearn version:", sklearn.__version__)
    joblib.dump(obj, path)
    print(f"💾 Saved {name} to cache → {path}")
def main():
    t0 = time.time()
    wide = cache_load("wide")
    if wide is None:
        print("Loading daily wide prices… (first time)")
        wide = load_wide_prices(DATA_FILE)
        cache_save("wide", wide)
    else:
        print("Using cached wide data")
    monthly_prices = cache_load("monthly_prices")
    weekly_prices  = cache_load("weekly_prices")
    daily_prices   = cache_load("daily_prices")

    if monthly_prices is None:
        print("Converting daily → monthly/weekly/daily (first time)")
        monthly_prices = wide_to_prices(wide, freq="ME")
        weekly_prices  = wide_to_prices(wide, freq="W")
        daily_prices   = wide_to_prices(wide, freq="D")

        cache_save("monthly_prices", monthly_prices)
        cache_save("weekly_prices", weekly_prices)
        cache_save("daily_prices",  daily_prices)
    else:
        print("Using cached monthly/weekly/daily")

    # -----------------------
    # 3) Load or build supervised arrays X, y, d
    # -----------------------
    supervised = cache_load("supervised_arrays_class")
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
        cache_save("supervised_arrays_class", (X, y, d, R, stocks))
        joblib.dump(stocks, "artifacts/stocks.joblib")
    else:
        print("Using cached supervised arrays (X, y, d)")
        X, y, d, R, stocks = supervised

    print(f"✅ Samples: {X.shape[0]:,} | Features: {X.shape[1]} "
          f"(M:{HISTORY_LAGS_M} + W:{HISTORY_LAGS_W} + D:{HISTORY_LAGS_D}) "
          f"| Positives: {y.sum():,} ({y.mean( ) *100:.2f}%)")

    # Optional: run hyperparameter search for a single frequency
    if RUN_SINGLE_SCALE_HYPEROPT:
        def hyperopt_single_scale(
                scale: str,
                lags_candidates,
                layer_configs,
                monthly_prices,
                weekly_prices,
                daily_prices,
        ):
            assert scale in {"monthly", "weekly", "daily"}
            best_config = None
            # We now maximize validation precision on class 1,
            # subject to at least 5% of predictions being 1 on the validation set.
            best_score = -np.inf
            best_summary = None
            all_results = []

            for lags_value in lags_candidates:
                # Decide triplet of lags for multiscale builder
                l_m, l_w, l_d = HISTORY_LAGS_M, HISTORY_LAGS_W, HISTORY_LAGS_D
                if scale == "monthly":
                    l_m = lags_value
                elif scale == "weekly":
                    l_w = lags_value
                else:
                    l_d = lags_value

                print(f"\n=== Building supervised arrays for {scale} with history {lags_value} ===")
                X_tmp, y_tmp, d_tmp, R_tmp, stocks_tmp = build_supervised_multiscale(
                    monthly=monthly_prices,
                    weekly=weekly_prices,
                    daily=daily_prices,
                    lags_m=l_m,
                    lags_w=l_w,
                    lags_d=l_d,
                )

                train_mask, val_mask_, test_mask_ = split_masks(
                    d_tmp, val_months=VAL_MONTHS, test_months=TEST_MONTHS
                )

                # Slice out the features for the selected scale
                if scale == "monthly":
                    X_scale = X_tmp[:, :l_m]
                elif scale == "weekly":
                    X_scale = X_tmp[:, l_m:l_m + l_w]
                else:  # daily
                    X_scale = X_tmp[:, l_m + l_w:]

                X_train, X_val = X_scale[train_mask], X_scale[val_mask_]
                y_train, y_val_ = y_tmp[train_mask], y_tmp[val_mask_]

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)

                for layers in layer_configs:
                    for s in ["adam"]:

                        lr_candidates = HYPEROPT_LEARNING_RATES if s in ["adam", "sgd"] else [None]

                        for lr_init in lr_candidates:
                            print(
                                f"\n--- {scale} | lags={lags_value} | layers={layers} | "
                                f"solver={s} | lr_init={lr_init} ---"
                            )

                            common_params = dict(
                                hidden_layer_sizes=layers,
                                activation="relu",
                                solver=s,
                                alpha=0.001,
                                max_iter=500,
                                random_state=42,
                                verbose=False,
                            )

                            if s == "adam":
                                model = MLPClassifier(
                                    **common_params,
                                    learning_rate_init=lr_init,
                                    early_stopping=True,
                                    n_iter_no_change=20,
                                    validation_fraction=0.10,
                                )

                            elif s == "sgd":
                                model = MLPClassifier(
                                    **common_params,
                                    learning_rate_init=lr_init,
                                    learning_rate="adaptive",
                                    early_stopping=True,
                                    n_iter_no_change=20,
                                    validation_fraction=0.10,
                                )

                            else:  # lbfgs
                                model = MLPClassifier(
                                    **common_params
                                )

                            model.fit(X_train, y_train)

                            # Training-loss diagnostics
                            if hasattr(model, "loss_curve_") and model.loss_curve_ is not None and len(
                                    model.loss_curve_) > 0:
                                loss_curve_last = float(model.loss_curve_[-1])
                                loss_curve_len = len(model.loss_curve_)
                            else:
                                loss_curve_last = np.nan
                                loss_curve_len = 0

                            loss_final = float(model.loss_) if hasattr(model, "loss_") else np.nan

                            # Validation probabilities
                            y_val_prob = model.predict_proba(X_val)[:, 1]

                            thresholds = np.arange(0.40, 0.60 + 1e-9, 0.02)
                            for thr in thresholds:
                                y_val_pred = (y_val_prob >= thr).astype(int)

                                coverage = (y_val_pred == 1).mean()
                                prec = precision_score(y_val_, y_val_pred, pos_label=1, zero_division=0)

                                print(
                                    f"    solver={s} | lr_init={lr_init} | thr={thr:.2f} | "
                                    f"loss_curve_last={loss_curve_last:.4f}, "
                                    f"loss_final={loss_final:.4f}, "
                                    f"precision_1={prec:.4f}, coverage(P(ŷ=1))={coverage:.4f}"
                                )

                                all_results.append({
                                    "scale": scale,
                                    "lags": lags_value,
                                    "layers": layers,
                                    "solver": s,
                                    "learning_rate_init": lr_init,
                                    "threshold": float(thr),
                                    "loss_curve_last": float(loss_curve_last) if not np.isnan(
                                        loss_curve_last) else np.nan,
                                    "loss_final": float(loss_final) if not np.isnan(loss_final) else np.nan,
                                    "loss_curve_len": int(loss_curve_len),
                                    "precision_1": float(prec),
                                    "coverage": float(coverage),
                                })

                                if coverage >= 0.05 and prec > best_score:
                                    best_score = prec
                                    best_config = {
                                        "scale": scale,
                                        "lags": lags_value,
                                        "layers": layers,
                                        "solver": s,
                                        "learning_rate_init": lr_init,
                                        "threshold": float(thr),
                                    }
                                    best_summary = {
                                        "loss_curve_last": loss_curve_last,
                                        "loss_final": loss_final,
                                        "loss_curve_len": loss_curve_len,
                                        "precision_1": prec,
                                        "coverage": coverage,
                                    }

            # Nicely formatted summary table of all tried configurations
            if all_results:
                print("\n=== Hyperopt summary (sorted by precision_1, desc) ===")
                all_results_sorted = sorted(
                    all_results,
                    key=lambda r: (r["precision_1"], r["coverage"]),
                    reverse=True,
                )

                print(
                    "scale  lags  layers                         solver  lr_init   thr   "
                    "precision_1  coverage  loss_curve_last  loss_final"
                )

                for r in all_results_sorted:
                    lr_text = "None" if r["learning_rate_init"] is None else f"{r['learning_rate_init']:.4g}"
                    print(
                        f"{r['scale']:6} "
                        f"{r['lags']:4d} "
                        f"{str(r['layers']):28} "
                        f"{r['solver']:6} "
                        f"{lr_text:8} "
                        f"{r['threshold']:5.2f} "
                        f"{r['precision_1']:11.4f} "
                        f"{r['coverage']:8.4f} "
                        f"{r['loss_curve_last']:15.4f} "
                        f"{r['loss_final']:10.4f}"
                    )

            print("\n=== Best hyperparameters for single-scale MLP (by precision_1 with coverage>=0.05) ===")
            print(best_config)
            print(best_summary)
            return best_config, best_summary

        # Run the hyperparameter search for the configured scale
        hyperopt_single_scale(
            scale=HYPEROPT_SCALE,
            lags_candidates=HYPEROPT_LAGS,
            layer_configs=HYPEROPT_LAYER_CONFIGS,
            monthly_prices=monthly_prices,
            weekly_prices=weekly_prices,
            daily_prices=daily_prices,
        )
        elapsed = time.time() - t0
        print(f"\n⏱️ Total runtime (with hyperopt): {elapsed /60:.2f} minutes ({elapsed:.1f} seconds)")
        return

    # Split into train/val/test (for the fixed multi-scale setup below)
    train_core_mask, val_mask, test_mask = split_masks(d, val_months=VAL_MONTHS, test_months=TEST_MONTHS)

    # --- Split features by scale ---
    lags_m, lags_w, lags_d = HISTORY_LAGS_M, HISTORY_LAGS_W, HISTORY_LAGS_D
    X_m = X[:, :lags_m]
    X_w = X[:, lags_m:lags_m +lags_w]
    X_d = X[:, lags_m +lags_w:]

    def train_and_get_probs(X_data, label):
        # Split data
        X_train, X_val, X_test = X_data[train_core_mask], X_data[val_mask], X_data[test_mask]
        y_train, y_val, y_test = y[train_core_mask], y[val_mask], y[test_mask]

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)
        X_test  = scaler.transform(X_test)

        # Define model
        model = MLPClassifier(
            hidden_layer_sizes=(512, 256, 128, 64),
            activation="relu",
            solver="adam",
            learning_rate_init=0.0005,
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

        # Get validation probabilities once
        y_prob_val = model.predict_proba(X_val)[:, 1]

        return {
            "model": model,
            "scaler": scaler,
            "y_prob_val": y_prob_val,
            "y_val": y_val,
        }


    y_val = y[val_mask]
    R = R[val_mask]
    # --- Train models once ---
    monthly_out = train_and_get_probs(X_m, "Monthly")
    weekly_out  = train_and_get_probs(X_w, "Weekly")
    daily_out   = train_and_get_probs(X_d, "Daily")

    y_prob_m = monthly_out["y_prob_val"]
    y_prob_w = weekly_out["y_prob_val"]
    y_prob_d = daily_out["y_prob_val"]

    def save_model_bundle(name, bundle):
        path = MODEL_DIR / f"{name}.joblib"
        joblib.dump(bundle, path)
        print(f"✅ Saved model artifact → {path}")

    # --- Test multiple thresholds efficiently ---
    thresholds = np.arange(0.50, 0.62 ,0.02)
    results = []
    for thr in thresholds:
        y_pred_m = (y_prob_m >= thr).astype(int)
        y_pred_w = (y_prob_w >= thr).astype(int)
        y_pred_d = (y_prob_d >= thr).astype(int)
        signal_mask = (
                (y_prob_m >= thr) &
                (y_prob_w >= thr) &
                (y_prob_d >= thr)
        )

        n_signals = signal_mask.sum()
        if n_signals == 0:
            continue
        realized_returns = R[signal_mask]
        # strict consensus (AND rule)
        y_pred_combined = ((y_pred_m == 1) & (y_pred_w == 1) & (y_pred_d == 1)).astype(int)

        print(f"\n=== RESULTS with threshold {thr:.2f} ===")
        print(classification_report(y_val, y_pred_combined, digits=4))
        # Optional: show agreement rate
        agreement = np.mean(y_pred_combined == y_val)
        print(f"✅ Agreement with true labels: {agreement *100:.2f}%")
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

    df_results = pd.DataFrame(results).sort_values("threshold").reset_index(drop=True)
    df_results.to_csv("df_results.csv", index=False)

    print("\n=== Threshold evaluation results ===")
    print(df_results)
    elapsed = time.time() - t0
    print(f"\n⏱️ Total runtime: {elapsed /60:.2f} minutes ({elapsed:.1f} seconds)")



    last_train_date = pd.to_datetime(d[train_core_mask]).max()

    save_model_bundle(
        "monthly_classification_v180",
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
        "weekly_classification_v180",
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
        "daily_classification_v180",
        {
            "model": daily_out["model"],
            "scaler": daily_out["scaler"],
            "lags": HISTORY_LAGS_D,
            "frequency": "daily",
            "trained_until": last_train_date,
            "sklearn_version": sklearn.__version__,
        },
    )
    print("models saved")

if __name__ == "__main__":
    main()