#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
import sklearn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import warnings
import json

print(f"Using sklearn {sklearn.__version__}")
warnings.filterwarnings("ignore", category=UserWarning)
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
DATA_FILE = 'nasdaq_exchange_daily_price_data_close.csv'
CUTOFF_DATE = pd.Timestamp("2000-01-01", tz="UTC")
HISTORY_LAGS_M = 150
HISTORY_LAGS_W = 150
HISTORY_LAGS_D = 150
VAL_MONTHS = 36
TEST_MONTHS = 36

# Hyperparameter search config for a single frequency
RUN_SINGLE_SCALE_HYPEROPT = True
HYPEROPT_SCALE = "weekly"  # one of {"monthly", "weekly", "daily"}
MODEL_OUT = f"mlp_{HYPEROPT_SCALE}_regression.joblib"
HYPEROPT_LAGS = [50, 100, 150]      #[50, 100, 150]
HYPEROPT_LAYER_CONFIGS = [
    (128,64),
    (32,16,8),
    (8, 4)
]
HYPEROPT_LEARNING_RATES = [0.0001, 0.0005, 0.001, 0.005]  #[0.0001, 0.0005, 0.001, 0.005]

# Threshold candidates: applied to predicted log-return
HYPEROPT_THRESHOLDS = np.arange(-0.01, 0.01, 0.0002)

MODEL_DIR = Path("artifacts")
MODEL_DIR.mkdir(exist_ok=True)

# -----------------------
# Helpers
# -----------------------
print("Made it")

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
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[df["date"] >= CUTOFF_DATE].reset_index(drop=True)
    return df

def wide_to_prices(df_wide: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
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
    """
    Build (X, y, dates, stocks) for regression.
    Features: same rolling multiscale log-return windows as the classifier.
    Label: next-month log-return (continuous float), not a binary direction.

    Returns
    -------
    X      : np.ndarray, shape (n_samples, lags_m + lags_w + lags_d), float32
    y      : np.ndarray, shape (n_samples,), float32  ← continuous log-return
    d      : np.ndarray, shape (n_samples,), datetime64
    stocks : np.ndarray, shape (n_samples,), object
    """
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
        m_ret   = gm["logret"].to_numpy()

        w_dates = gw["date"].to_numpy()
        w_ret   = gw["logret"].to_numpy()

        d_dates = gd["date"].to_numpy()
        d_ret   = gd["logret"].to_numpy()

        for t in range(1, len(gm) - 1):
            t_date = m_dates[t]

            if t < lags_m:
                continue
            m_window = m_ret[t - lags_m + 1: t + 1]
            if np.any(np.isnan(m_window)) or len(m_window) != lags_m:
                continue

            idx_w_end = np.searchsorted(w_dates, t_date, side="right") - 1
            if idx_w_end < 0 or idx_w_end + 1 < lags_w:
                continue
            w_window = w_ret[idx_w_end - lags_w + 1: idx_w_end + 1]
            if np.any(np.isnan(w_window)) or len(w_window) != lags_w:
                continue

            idx_d_end = np.searchsorted(d_dates, t_date, side="right") - 1
            if idx_d_end < 0 or idx_d_end + 1 < lags_d:
                continue
            d_window = d_ret[idx_d_end - lags_d + 1: idx_d_end + 1]
            if np.any(np.isnan(d_window)) or len(d_window) != lags_d:
                continue

            # Regression label: next-month log-return (continuous)
            label = m_ret[t + 1]
            if np.isnan(label):
                continue

            x = np.concatenate([m_window, w_window, d_window], dtype=np.float32)
            X_list.append(x)
            Y_list.append(label)
            D_list.append(t_date)
            Stock_list.append(s)

    if not X_list:
        raise ValueError("No samples built. Check lags and that all three frequencies have sufficient history.")

    X      = np.vstack(X_list).astype(np.float32)
    y      = np.asarray(Y_list, dtype=np.float32)
    d_out  = np.asarray(D_list)
    stocks = np.asarray(Stock_list, dtype=object)
    return X, y, d_out, stocks


def split_masks(dates: np.ndarray, val_months: int = VAL_MONTHS, test_months: int = TEST_MONTHS):
    dates = pd.to_datetime(dates)
    if getattr(dates, "tz", None) is not None:
        dates = dates.tz_localize(None)

    last_date  = dates.max()
    train_end  = last_date - pd.DateOffset(months=(val_months + test_months))
    val_start  = train_end  + pd.DateOffset(months=1)
    val_end    = last_date  - pd.DateOffset(months=test_months)
    test_start = val_end    + pd.DateOffset(months=1)
    test_end   = last_date

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
    path = CACHE_DIR / f"{name}.joblib"
    print("SAVE sklearn version:", sklearn.__version__)
    joblib.dump(obj, path)
    print(f"💾 Saved {name} to cache → {path}")

def save_model_bundle(name, bundle):
    path = MODEL_DIR / f"{name}.joblib"
    joblib.dump(bundle, path)
    print(f"✅ Saved model artifact → {path}")


def main():
    t0 = time.time()

    # -----------------------
    # 1) Load raw price data
    # -----------------------
    wide = cache_load("wide")
    if wide is None:
        print("Loading daily wide prices… (first time)")
        wide = load_wide_prices(DATA_FILE)
        cache_save("wide", wide)
    else:
        print("Using cached wide data")

    # -----------------------
    # 2) Resample to monthly / weekly / daily
    # -----------------------
    monthly_prices = cache_load("monthly_prices")
    weekly_prices  = cache_load("weekly_prices")
    daily_prices   = cache_load("daily_prices")

    if monthly_prices is None:
        print("Converting daily → monthly/weekly/daily (first time)")
        monthly_prices = wide_to_prices(wide, freq="ME")
        weekly_prices  = wide_to_prices(wide, freq="W")
        daily_prices   = wide_to_prices(wide, freq="D")
        cache_save("monthly_prices", monthly_prices)
        cache_save("weekly_prices",  weekly_prices)
        cache_save("daily_prices",   daily_prices)
    else:
        print("Using cached monthly/weekly/daily")

    # -----------------------
    # 3) Build supervised arrays (regression version)
    # -----------------------
    supervised = cache_load("supervised_arrays_regression")
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
        cache_save("supervised_arrays_regression", (X, y, d, stocks))
        joblib.dump(stocks, "artifacts/stocks.joblib")
    else:
        print("Using cached supervised arrays (X, y, d, stocks)")
        X, y, d, stocks = supervised

    print(f"✅ Samples: {X.shape[0]:,} | Features: {X.shape[1]} "
          f"(M:{HISTORY_LAGS_M} + W:{HISTORY_LAGS_W} + D:{HISTORY_LAGS_D}) "
          f"| Mean target return: {y.mean():.5f} | Std: {y.std():.5f}")

    # -----------------------
    # 4) Hyperparameter search (single scale, regression)
    # -----------------------
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
            best_config  = None
            best_score   = -np.inf   # maximize hit_rate subject to coverage >= 0.05
            best_summary = None
            all_results  = []

            for lags_value in lags_candidates:
                l_m, l_w, l_d = HISTORY_LAGS_M, HISTORY_LAGS_W, HISTORY_LAGS_D
                if scale == "monthly":
                    l_m = lags_value
                elif scale == "weekly":
                    l_w = lags_value
                else:
                    l_d = lags_value

                print(f"\n=== Building supervised arrays for {scale} with history {lags_value} ===")
                X_tmp, y_tmp, d_tmp, stocks_tmp = build_supervised_multiscale(
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

                # Slice out features for the selected scale only
                if scale == "monthly":
                    X_scale = X_tmp[:, :l_m]
                elif scale == "weekly":
                    X_scale = X_tmp[:, l_m:l_m + l_w]
                else:
                    X_scale = X_tmp[:, l_m + l_w:]

                X_train, X_val = X_scale[train_mask], X_scale[val_mask_]
                y_train, y_val_ = y_tmp[train_mask], y_tmp[val_mask_]

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val   = scaler.transform(X_val)

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
                                model = MLPRegressor(
                                    **common_params,
                                    learning_rate_init=lr_init,
                                    early_stopping=True,
                                    n_iter_no_change=20,
                                    validation_fraction=0.10,
                                )
                            elif s == "sgd":
                                model = MLPRegressor(
                                    **common_params,
                                    learning_rate_init=lr_init,
                                    learning_rate="adaptive",
                                    early_stopping=True,
                                    n_iter_no_change=20,
                                    validation_fraction=0.10,
                                )
                            else:  # lbfgs
                                model = MLPRegressor(**common_params)

                            model.fit(X_train, y_train)

                            # Training-loss diagnostics
                            if hasattr(model, "loss_curve_") and model.loss_curve_ is not None and len(
                                    model.loss_curve_) > 0:
                                loss_curve_last = float(model.loss_curve_[-1])
                                loss_curve_len  = len(model.loss_curve_)
                            else:
                                loss_curve_last = np.nan
                                loss_curve_len  = 0

                            loss_final = float(model.loss_) if hasattr(model, "loss_") else np.nan

                            # Validation predictions (continuous log-returns)
                            y_val_pred = model.predict(X_val)

                            # Sweep thresholds: signal = predicted return >= thr
                            for thr in HYPEROPT_THRESHOLDS:
                                signal_mask = y_val_pred >= thr
                                coverage    = signal_mask.mean()

                                if coverage == 0:
                                    hit_rate    = 0.0
                                    mean_return = np.nan
                                else:
                                    realized    = y_val_[signal_mask]
                                    hit_rate    = float(np.mean(realized >= 0)) 
                                    mean_return = float(np.mean(realized))

                                print(
                                    f"    solver={s} | lr_init={lr_init} | thr={thr:.4f} | "
                                    f"loss_curve_last={loss_curve_last:.4f}, "
                                    f"loss_final={loss_final:.4f}, "
                                    f"hit_rate={hit_rate:.4f}, coverage={coverage:.4f}, "
                                    f"mean_return={mean_return:.5f}"
                                )

                                all_results.append({
                                    "scale":              scale,
                                    "lags":               lags_value,
                                    "layers":             layers,
                                    "solver":             s,
                                    "learning_rate_init": lr_init,
                                    "threshold":          float(thr),
                                    "loss_curve_last":    float(loss_curve_last) if not np.isnan(loss_curve_last) else np.nan,
                                    "loss_final":         float(loss_final)      if not np.isnan(loss_final)      else np.nan,
                                    "loss_curve_len":     int(loss_curve_len),
                                    "hit_rate":           float(hit_rate),
                                    "coverage":           float(coverage),
                                    "mean_return":        float(mean_return) if not np.isnan(mean_return) else np.nan,
                                })

                                if coverage >= 0.05 and hit_rate > best_score:
                                    best_score = hit_rate
                                    best_config = {
                                        "scale":              scale,
                                        "lags":               lags_value,
                                        "layers":             layers,
                                        "solver":             s,
                                        "learning_rate_init": lr_init,
                                        "threshold":          float(thr),
                                    }
                                    best_summary = {
                                        "loss_curve_last": loss_curve_last,
                                        "loss_final":      loss_final,
                                        "loss_curve_len":  loss_curve_len,
                                        "hit_rate":        hit_rate,
                                        "coverage":        coverage,
                                        "mean_return":     mean_return,
                                    }

            # Summary table sorted by hit_rate
            if all_results:
                print("\n=== Hyperopt summary (sorted by hit_rate, desc) ===")
                all_results_sorted = sorted(
                    all_results,
                    key=lambda r: (r["hit_rate"], r["coverage"]),
                    reverse=True,
                )

                print(
                    "scale  lags  layers                         solver  lr_init   thr      "
                    "hit_rate  coverage  mean_return  loss_curve_last  loss_final"
                )

                for r in all_results_sorted:
                    lr_text = "None" if r["learning_rate_init"] is None else f"{r['learning_rate_init']:.4g}"
                    mr_text = f"{r['mean_return']:.5f}" if not np.isnan(r["mean_return"]) else "   nan"
                    print(
                        f"{r['scale']:6} "
                        f"{r['lags']:4d} "
                        f"{str(r['layers']):28} "
                        f"{r['solver']:6} "
                        f"{lr_text:8} "
                        f"{r['threshold']:7.4f} "
                        f"{r['hit_rate']:8.4f} "
                        f"{r['coverage']:8.4f} "
                        f"{mr_text:11} "
                        f"{r['loss_curve_last']:15.4f} "
                        f"{r['loss_final']:10.4f}"
                    )

            print("\n=== Best hyperparameters for single-scale MLP regressor (by hit_rate with coverage>=0.05) ===")
            print(best_config)
            print(best_summary)
            return best_config, best_summary
        print("Start good")
        hyperopt_single_scale(
            scale=HYPEROPT_SCALE,
            lags_candidates=HYPEROPT_LAGS,
            layer_configs=HYPEROPT_LAYER_CONFIGS,
            monthly_prices=monthly_prices,
            weekly_prices=weekly_prices,
            daily_prices=daily_prices,
        )
        print("End good")
        elapsed = time.time() - t0
        print(f"\n⏱️ Total runtime (with hyperopt): {elapsed / 60:.2f} minutes ({elapsed:.1f} seconds)")
        return
    
    # -----------------------
    # 5) Fixed multi-scale training (runs when RUN_SINGLE_SCALE_HYPEROPT = False)
    # -----------------------
    train_core_mask, val_mask, test_mask = split_masks(d, val_months=VAL_MONTHS, test_months=TEST_MONTHS)

    lags_m, lags_w, lags_d = HISTORY_LAGS_M, HISTORY_LAGS_W, HISTORY_LAGS_D
    X_m = X[:, :lags_m]
    X_w = X[:, lags_m:lags_m + lags_w]
    X_d = X[:, lags_m + lags_w:]

    def train_and_get_preds(X_data, label):
        X_train = X_data[train_core_mask]
        X_val   = X_data[val_mask]
        X_test  = X_data[test_mask]
        y_train = y[train_core_mask]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)
        X_test  = scaler.transform(X_test)

        model = MLPRegressor(
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

        y_pred_val = model.predict(X_val)
        return {
            "model":      model,
            "scaler":     scaler,
            "y_pred_val": y_pred_val,
        }

    y_val = y[val_mask]

    monthly_out = train_and_get_preds(X_m, "Monthly")
    weekly_out  = train_and_get_preds(X_w, "Weekly")
    daily_out   = train_and_get_preds(X_d, "Daily")

    y_pred_m = monthly_out["y_pred_val"]
    y_pred_w = weekly_out["y_pred_val"]
    y_pred_d = daily_out["y_pred_val"]

    # --- Threshold sweep on validation ---
    thresholds = np.arange(0.002, 0.01, 0.0002)
    results = []

    for thr in thresholds:
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
            "threshold":     round(float(thr), 4),
            "coverage":      n_signals / len(y_val),
            "n_signals":     n_signals,
            "hit_rate":      float(np.mean(realized_returns >= 0)),
            "mean_return":   float(np.mean(realized_returns)),
            "median_return": float(np.median(realized_returns)),
            "variance":      float(np.var(realized_returns)),
            "min_return":    float(np.min(realized_returns)),
            "max_return":    float(np.max(realized_returns)),
            "quantile_5":    float(np.percentile(realized_returns, 5)),
            "quantile_10":   float(np.percentile(realized_returns, 10)),
            "quantile_90":   float(np.percentile(realized_returns, 90)),
            "quantile_95":   float(np.percentile(realized_returns, 95)),
        })

    df_results = pd.DataFrame(results).sort_values("threshold").reset_index(drop=True)
    df_results.to_csv("df_results_regression.csv", index=False)

    print("\n=== Threshold evaluation results ===")
    print(df_results)
    elapsed = time.time() - t0
    print(f"\n⏱️ Total runtime: {elapsed / 60:.2f} minutes ({elapsed:.1f} seconds)")

    last_train_date = pd.to_datetime(d[train_core_mask]).max()

    save_model_bundle(
        "monthly_regression_v180",
        {
            "model":           monthly_out["model"],
            "scaler":          monthly_out["scaler"],
            "lags":            HISTORY_LAGS_M,
            "frequency":       "monthly",
            "trained_until":   last_train_date,
            "sklearn_version": sklearn.__version__,
        },
    )
    save_model_bundle(
        "weekly_regression_v180",
        {
            "model":           weekly_out["model"],
            "scaler":          weekly_out["scaler"],
            "lags":            HISTORY_LAGS_W,
            "frequency":       "weekly",
            "trained_until":   last_train_date,
            "sklearn_version": sklearn.__version__,
        },
    )
    save_model_bundle(
        "daily_regression_v180",
        {
            "model":           daily_out["model"],
            "scaler":          daily_out["scaler"],
            "lags":            HISTORY_LAGS_D,
            "frequency":       "daily",
            "trained_until":   last_train_date,
            "sklearn_version": sklearn.__version__,
        },
    )
    print("models saved")


if __name__ == "__main__":
    main()