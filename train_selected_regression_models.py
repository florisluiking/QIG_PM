#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

print(f"Using sklearn {sklearn.__version__}")

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------
# Reproducibility / threads
# -----------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTHONHASHSEED"] = "0"

np.random.seed(42)

# -----------------------
# Config
# -----------------------

DATA_FILE = "nasdaq_exchange_daily_price_data_close.csv"

# Raw data cutoff.
# This only controls which raw data is loaded.
# It is NOT the final training period.
CUTOFF_DATE = pd.Timestamp("1980-01-01")

# Default lags used as placeholders.
# For each selected model, only the relevant scale lag is overwritten.
HISTORY_LAGS_M = 150
HISTORY_LAGS_W = 150
HISTORY_LAGS_D = 150

# Final training period.
# These are timezone-naive on purpose because all dates are converted to timezone-naive.
FINAL_TRAIN_START = pd.Timestamp("2000-01-01")
FINAL_TRAIN_END = pd.Timestamp("2021-12-31")

MODEL_DIR = Path("artifacts")
MODEL_DIR.mkdir(exist_ok=True)

CACHE_DIR = Path("cache_simple")
CACHE_DIR.mkdir(exist_ok=True)


# -----------------------
# Helpers
# -----------------------

def find_first_datetime_row(csv_path: str) -> int:
    raw = pd.read_csv(csv_path, header=None, nrows=500, low_memory=False)

    for idx, val in enumerate(raw.iloc[:, 0]):
        try:
            pd.to_datetime(val)
            return idx
        except Exception:
            continue

    return 0


def load_wide_prices(csv_path: str) -> pd.DataFrame:
    first_data_row = find_first_datetime_row(csv_path)

    df = pd.read_csv(csv_path, skiprows=range(first_data_row), low_memory=False)
    df.rename(columns={df.columns[0]: "date"}, inplace=True)

    # Strong timezone fix:
    # 1. Parse as UTC so mixed formats are handled consistently.
    # 2. Remove timezone so all comparisons are timezone-naive.
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["date"] = df["date"].dt.tz_localize(None)

    df = df.dropna(subset=["date"])

    # Keep only data from 1980 onward.
    df = df[df["date"] >= CUTOFF_DATE].reset_index(drop=True)

    return df


def wide_to_prices(df_wide: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    df_wide = df_wide.copy()

    # Extra safety in case cached or loaded data still has timezone-aware dates.
    df_wide["date"] = pd.to_datetime(df_wide["date"], errors="coerce", utc=True)
    df_wide["date"] = df_wide["date"].dt.tz_localize(None)

    long_df = pd.melt(
        df_wide,
        id_vars=["date"],
        var_name="stock",
        value_name="price"
    )

    long_df["price"] = pd.to_numeric(long_df["price"], errors="coerce")
    long_df = long_df.dropna(subset=["price"])
    long_df = long_df[long_df["price"] > 0]

    long_df["date"] = pd.to_datetime(long_df["date"], errors="coerce", utc=True)
    long_df["date"] = long_df["date"].dt.tz_localize(None)

    def resample_prices(g, freq):
        g = g.sort_values("date").set_index("date")

        if freq == "D":
            out = g[["price"]].reset_index()
        else:
            out = g["price"].resample(freq).last().dropna().reset_index()

        out["stock"] = g["stock"].iloc[0]
        return out

    resampled_list = [
        resample_prices(g, freq)
        for _, g in long_df.groupby("stock", sort=False)
    ]

    final = pd.concat(resampled_list, ignore_index=True)
    final = final.dropna(subset=["price"])
    final = final.sort_values(["stock", "date"]).reset_index(drop=True)

    # Extra safety after resampling.
    final["date"] = pd.to_datetime(final["date"], errors="coerce", utc=True)
    final["date"] = final["date"].dt.tz_localize(None)

    return final


def build_supervised_multiscale(
    monthly: pd.DataFrame,
    weekly: pd.DataFrame,
    daily: pd.DataFrame,
    lags_m: int = 100,
    lags_w: int = 100,
    lags_d: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build X, y, dates, stocks for regression.

    Features:
        [monthly log-return window | weekly log-return window | daily log-return window]

    Label:
        next-month log-return.
    """

    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # Strong timezone fix:
        # parse as UTC, then remove timezone.
        out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=True)
        out["date"] = out["date"].dt.tz_localize(None)

        out["price"] = pd.to_numeric(out["price"], errors="coerce")

        out = out.dropna(subset=["date", "price"])
        out = out.sort_values(["stock", "date"]).reset_index(drop=True)

        out["logret"] = (
            out.groupby("stock", sort=False)["price"]
            .transform(lambda p: np.log(p).diff())
        )

        return out

    m = _prep(monthly)
    w = _prep(weekly)
    dly = _prep(daily)

    stocks = sorted(set(m["stock"]))

    X_list = []
    Y_list = []
    D_list = []
    Stock_list = []

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

            # Monthly window.
            if t < lags_m:
                continue

            m_window = m_ret[t - lags_m + 1: t + 1]

            if np.any(np.isnan(m_window)) or len(m_window) != lags_m:
                continue

            # Weekly window up to current monthly date.
            idx_w_end = np.searchsorted(w_dates, t_date, side="right") - 1

            if idx_w_end < 0 or idx_w_end + 1 < lags_w:
                continue

            w_window = w_ret[idx_w_end - lags_w + 1: idx_w_end + 1]

            if np.any(np.isnan(w_window)) or len(w_window) != lags_w:
                continue

            # Daily window up to current monthly date.
            idx_d_end = np.searchsorted(d_dates, t_date, side="right") - 1

            if idx_d_end < 0 or idx_d_end + 1 < lags_d:
                continue

            d_window = d_ret[idx_d_end - lags_d + 1: idx_d_end + 1]

            if np.any(np.isnan(d_window)) or len(d_window) != lags_d:
                continue

            # Regression label: next-month log-return.
            label = m_ret[t + 1]

            if np.isnan(label):
                continue

            x = np.concatenate([m_window, w_window, d_window]).astype(np.float32)

            X_list.append(x)
            Y_list.append(label)
            D_list.append(t_date)
            Stock_list.append(s)

    if not X_list:
        raise ValueError(
            "No samples built. Check lags and whether all frequencies have enough history."
        )

    X = np.vstack(X_list).astype(np.float32)
    y = np.asarray(Y_list, dtype=np.float32)

    # Important: convert output dates to timezone-naive datetime64.
    d_out = pd.to_datetime(np.asarray(D_list), utc=True).tz_localize(None).to_numpy()

    stocks_out = np.asarray(Stock_list, dtype=object)

    return X, y, d_out, stocks_out


def cache_load(name):
    path = CACHE_DIR / f"{name}.joblib"

    if path.exists():
        print(f"🔄 Loading cached {name} from {path}")
        return joblib.load(path)

    return None


def cache_save(name, obj):
    path = CACHE_DIR / f"{name}.joblib"

    print("SAVE sklearn version:", sklearn.__version__)
    joblib.dump(obj, path)

    print(f"💾 Saved {name} to cache → {path}")


def save_model_bundle(name, bundle):
    path = MODEL_DIR / f"{name}.joblib"

    joblib.dump(bundle, path)

    print(f"✅ Saved model artifact → {path}")


# -----------------------
# Selected final regression models
# -----------------------
# Change only this section if your selected final models are different.

SELECTED_MODELS = [
    {
        "name": "daily_regression_lags50_layers128_64_adam_lr00005_thr0006",
        "scale": "daily",
        "lags": 50,
        "layers": (128, 64),
        "solver": "adam",
        "learning_rate_init": 0.0005,
        "threshold": None,
    },
    {
        "name": "weekly_regression_lags150_layers8_4_adam_lr0005",
        "scale": "weekly",
        "lags": 100,
        "layers": (32, 16, 8),
        "solver": "adam",
        "learning_rate_init": 0.005,
        "threshold": None,
    },
    {
        "name": "monthly_regression_lags150_layers8_4_adam_lr0005_thr0007",
        "scale": "monthly",
        "lags": 150,
        "layers": (8, 4),
        "solver": "adam",
        "learning_rate_init": 0.005,
        "threshold": None,
    },
]


def train_selected_single_scale_model(
    config,
    monthly_prices,
    weekly_prices,
    daily_prices,
):
    scale = config["scale"]
    lags_value = config["lags"]

    # Start from default lags.
    # Then replace only the scale we are training.
    l_m = HISTORY_LAGS_M
    l_w = HISTORY_LAGS_W
    l_d = HISTORY_LAGS_D

    if scale == "monthly":
        l_m = lags_value
    elif scale == "weekly":
        l_w = lags_value
    elif scale == "daily":
        l_d = lags_value
    else:
        raise ValueError(f"Unknown scale: {scale}")

    print("\n==============================")
    print(f"Training final {scale} regression model")
    print(f"Lags: {lags_value}")
    print(f"Layers: {config['layers']}")
    print(f"Solver: {config['solver']}")
    print(f"Learning rate: {config['learning_rate_init']}")
    print(f"Threshold: {config.get('threshold', None)}")
    print("==============================")

    # Build supervised data with correct lag length.
    # X_tmp contains: [monthly features | weekly features | daily features]
    X_tmp, y_tmp, d_tmp, stocks_tmp = build_supervised_multiscale(
        monthly=monthly_prices,
        weekly=weekly_prices,
        daily=daily_prices,
        lags_m=l_m,
        lags_w=l_w,
        lags_d=l_d,
    )

    # Strong timezone fix for training/test masks.
    dates_tmp = pd.to_datetime(d_tmp, utc=True).tz_localize(None)

    # Fixed final training period: 2000 to end of 2021.
    train_final_mask = (
        (dates_tmp >= FINAL_TRAIN_START) &
        (dates_tmp <= FINAL_TRAIN_END)
    )

    # Out-of-sample period after 2021.
    test_mask = dates_tmp > FINAL_TRAIN_END

    print("\nFixed final training period:")
    print(
        f"  Train: {FINAL_TRAIN_START.date()} → {FINAL_TRAIN_END.date()} "
        f"({train_final_mask.sum()} samples)"
    )
    print(
        f"  Test/backtest after: {FINAL_TRAIN_END.date()} "
        f"({test_mask.sum()} samples)"
    )

    if train_final_mask.sum() == 0:
        raise ValueError(
            f"No training samples found for {scale} between "
            f"{FINAL_TRAIN_START.date()} and {FINAL_TRAIN_END.date()}."
        )

    # Select only the feature block for the chosen scale.
    if scale == "monthly":
        X_scale = X_tmp[:, :l_m]
    elif scale == "weekly":
        X_scale = X_tmp[:, l_m:l_m + l_w]
    else:
        X_scale = X_tmp[:, l_m + l_w:]

    X_train = X_scale[train_final_mask]
    y_train = y_tmp[train_final_mask]

    X_test = X_scale[test_mask]
    y_test = y_tmp[test_mask]

    # Fit scaler only on final training period.
    # This prevents data leakage from the post-2021 backtest period.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    if test_mask.sum() > 0:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = None

    solver = config["solver"]

    common_params = dict(
        hidden_layer_sizes=config["layers"],
        activation="relu",
        solver=solver,
        alpha=0.001,
        max_iter=500,
        random_state=42,
        verbose=True,
    )

    if solver == "adam":
        model = MLPRegressor(
            **common_params,
            learning_rate_init=config["learning_rate_init"],
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.10,
        )
    else:
        raise ValueError(
            "This final selected-model script expects all models to use solver='adam'."
        )

    print(f"\n🧠 Fitting {scale} model...")
    model.fit(X_train_scaled, y_train)

    # Sanity check on post-2021 period, if available.
    if test_mask.sum() > 0:
        y_test_pred = model.predict(X_test_scaled)

        print(f"\nSanity check for {scale} model:")
        print(f"  Test predictions mean: {np.mean(y_test_pred):.6f}")
        print(f"  Test predictions std:  {np.std(y_test_pred):.6f}")
        print(f"  Test realized mean:    {np.mean(y_test):.6f}")

        if config.get("threshold", None) is not None:
            threshold = config["threshold"]
            signal_mask = y_test_pred >= threshold
            coverage = signal_mask.mean()

            if signal_mask.sum() > 0:
                realized_signal_returns = y_test[signal_mask]
                hit_rate = np.mean(realized_signal_returns >= 0)
                mean_signal_return = np.mean(realized_signal_returns)
            else:
                hit_rate = np.nan
                mean_signal_return = np.nan

            print(f"\nThreshold sanity check for {scale} model:")
            print(f"  Threshold:          {threshold:.4f}")
            print(f"  Signal coverage:    {coverage:.4f}")
            print(f"  Signal hit rate:    {hit_rate:.4f}")
            print(f"  Mean signal return: {mean_signal_return:.6f}")

    else:
        print(f"\nNo post-2021 test samples available for {scale} sanity check.")

    # Save model bundle.
    bundle = {
        # Core objects needed for backtesting.
        "model": model,
        "scaler": scaler,

        # Model identity.
        "scale": scale,
        "frequency": scale,
        "lags": lags_value,
        "layers": config["layers"],
        "solver": solver,
        "learning_rate_init": config["learning_rate_init"],

        # Full lag structure used when building X_tmp.
        "lags_m": l_m,
        "lags_w": l_w,
        "lags_d": l_d,

        # Threshold selected during validation/hyperopt.
        "threshold": config.get("threshold", None),

        # Training metadata.
        "trained_from": FINAL_TRAIN_START,
        "trained_until": FINAL_TRAIN_END,
        "sklearn_version": sklearn.__version__,

        # Useful metadata for checking the backtest period.
        "test_start": dates_tmp[test_mask].min() if test_mask.sum() > 0 else None,
        "test_end": dates_tmp[test_mask].max() if test_mask.sum() > 0 else None,
    }

    save_model_bundle(config["name"], bundle)

    # Save stock ordering for this model's supervised dataset.
    joblib.dump(stocks_tmp, MODEL_DIR / f"{config['name']}_stocks.joblib")

    return bundle


def main():
    t0 = time.time()

    # -----------------------
    # 1) Load raw price data
    # -----------------------
    wide = cache_load("wide")

    if wide is None:
        print("Loading daily wide prices... first time")
        wide = load_wide_prices(DATA_FILE)
        cache_save("wide", wide)
    else:
        print("Using cached wide data")

        # Extra safety: fix cached wide dates.
        wide["date"] = pd.to_datetime(wide["date"], errors="coerce", utc=True)
        wide["date"] = wide["date"].dt.tz_localize(None)

    # -----------------------
    # 2) Resample to monthly / weekly / daily
    # -----------------------
    monthly_prices = cache_load("monthly_prices")
    weekly_prices = cache_load("weekly_prices")
    daily_prices = cache_load("daily_prices")

    if monthly_prices is None or weekly_prices is None or daily_prices is None:
        print("Converting daily prices to monthly/weekly/daily... first time")

        monthly_prices = wide_to_prices(wide, freq="ME")
        weekly_prices = wide_to_prices(wide, freq="W")
        daily_prices = wide_to_prices(wide, freq="D")

        cache_save("monthly_prices", monthly_prices)
        cache_save("weekly_prices", weekly_prices)
        cache_save("daily_prices", daily_prices)

    else:
        print("Using cached monthly/weekly/daily prices")

        # Extra safety: fix cached resampled dates.
        for df_name, df_prices in [
            ("monthly_prices", monthly_prices),
            ("weekly_prices", weekly_prices),
            ("daily_prices", daily_prices),
        ]:
            df_prices["date"] = pd.to_datetime(df_prices["date"], errors="coerce", utc=True)
            df_prices["date"] = df_prices["date"].dt.tz_localize(None)
            print(f"Fixed timezone format for cached {df_name}")

    # -----------------------
    # 3) Train and save selected final models
    # -----------------------
    saved_bundles = []

    for config in SELECTED_MODELS:
        bundle = train_selected_single_scale_model(
            config=config,
            monthly_prices=monthly_prices,
            weekly_prices=weekly_prices,
            daily_prices=daily_prices,
        )

        saved_bundles.append(bundle)

    print("\n✅ Saved all selected regression models for backtesting:")

    for config in SELECTED_MODELS:
        print(f"  artifacts/{config['name']}.joblib")

    elapsed = time.time() - t0

    print(f"\n⏱️ Total runtime: {elapsed / 60:.2f} minutes ({elapsed:.1f} seconds)")


if __name__ == "__main__":
    main()