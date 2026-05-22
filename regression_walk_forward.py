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

# Reproducibility / thread control
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(42)

# -----------------------
# Paths
# -----------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "nasdaq_exchange_daily_price_data_close.csv"

CACHE_DIR = BASE_DIR / "cache_simple"
CACHE_DIR.mkdir(exist_ok=True)

MODEL_DIR = BASE_DIR / "artifacts"
MODEL_DIR.mkdir(exist_ok=True)

# -----------------------
# Config
# -----------------------

CUTOFF_DATE = pd.Timestamp("2000-01-01", tz="UTC")

HISTORY_LAGS_M = 150
HISTORY_LAGS_W = 150
HISTORY_LAGS_D = 150

VAL_MONTHS = 36
TEST_MONTHS = 36

# Hyperparameter search config for a single frequency
RUN_SINGLE_SCALE_HYPEROPT = True

# Change this to "monthly", "weekly", or "daily"
HYPEROPT_SCALE = "daily"

MODEL_OUT = f"mlp_{HYPEROPT_SCALE}_regression.joblib"

HYPEROPT_LAGS = [50, 100, 150]

HYPEROPT_LAYER_CONFIGS = [
    (128, 64),
    (32, 16, 8),
    (8, 4),
]

HYPEROPT_LEARNING_RATES = [0.0001, 0.0005, 0.001, 0.005]

# Threshold candidates applied to predicted log-return
HYPEROPT_THRESHOLDS = np.arange(-0.005, 0.010, 0.001)

# -----------------------
# Walk-forward validation settings
# -----------------------

USE_WALK_FORWARD = True

WF_VAL_MONTHS = 36          # each validation window = 3 years
WF_STEP_MONTHS = 36         # move forward by 3 years
WF_MIN_TRAIN_MONTHS = 156   # first train period approximately 2000–2012
WF_FINAL_TEST_MONTHS = 36   # final 3 years untouched for final backtest

print("Made it")


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
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"])

    df = df[df["date"] >= CUTOFF_DATE].reset_index(drop=True)

    return df


def wide_to_prices(df_wide: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    long_df = pd.melt(
        df_wide,
        id_vars=["date"],
        var_name="stock",
        value_name="price",
    )

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
        resample_prices(g, freq)
        for _, g in long_df.groupby("stock", sort=False)
    ]

    final = pd.concat(resampled_list, ignore_index=True)
    final = final.dropna(subset=["price"])
    final = final.sort_values(["stock", "date"]).reset_index(drop=True)

    return final


def build_supervised_multiscale(
    monthly: pd.DataFrame,
    weekly: pd.DataFrame,
    daily: pd.DataFrame,
    lags_m: int = 100,
    lags_w: int = 100,
    lags_d: int = 100,
):
    """
    Build X, y, dates, stocks for regression.

    Features:
    - monthly log-return window
    - weekly log-return window
    - daily log-return window

    Label:
    - next-month log return
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

        out["logret"] = out.groupby("stock", sort=False)["price"].transform(
            lambda p: np.log(p).diff()
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

            # Monthly window
            if t < lags_m:
                continue

            m_window = m_ret[t - lags_m + 1 : t + 1]

            if np.any(np.isnan(m_window)) or len(m_window) != lags_m:
                continue

            # Weekly window
            idx_w_end = np.searchsorted(w_dates, t_date, side="right") - 1

            if idx_w_end < 0 or idx_w_end + 1 < lags_w:
                continue

            w_window = w_ret[idx_w_end - lags_w + 1 : idx_w_end + 1]

            if np.any(np.isnan(w_window)) or len(w_window) != lags_w:
                continue

            # Daily window
            idx_d_end = np.searchsorted(d_dates, t_date, side="right") - 1

            if idx_d_end < 0 or idx_d_end + 1 < lags_d:
                continue

            d_window = d_ret[idx_d_end - lags_d + 1 : idx_d_end + 1]

            if np.any(np.isnan(d_window)) or len(d_window) != lags_d:
                continue

            # Regression label: next-month log return
            label = m_ret[t + 1]

            if np.isnan(label):
                continue

            x = np.concatenate([m_window, w_window, d_window], dtype=np.float32)

            X_list.append(x.astype(np.float32))
            Y_list.append(label)
            D_list.append(t_date)
            Stock_list.append(s)

    if not X_list:
        raise ValueError("No samples built. Check lags and data availability.")

    X = np.vstack(X_list).astype(np.float32)
    y = np.asarray(Y_list, dtype=np.float32)
    d_out = np.asarray(D_list)
    stocks_out = np.asarray(Stock_list, dtype=object)

    return X, y, d_out, stocks_out


def split_masks(dates: np.ndarray, val_months: int = VAL_MONTHS, test_months: int = TEST_MONTHS):
    """
    Standard single train-validation-test split.
    """

    dates = pd.to_datetime(dates)

    if getattr(dates, "tz", None) is not None:
        dates = dates.tz_localize(None)

    last_date = dates.max()

    train_end = last_date - pd.DateOffset(months=(val_months + test_months))
    val_start = train_end + pd.DateOffset(months=1)
    val_end = last_date - pd.DateOffset(months=test_months)
    test_start = val_end + pd.DateOffset(months=1)
    test_end = last_date

    train_core_mask = dates <= train_end
    val_mask = (dates >= val_start) & (dates <= val_end)
    test_mask = (dates >= test_start) & (dates <= test_end)

    print("Dynamic split dates:")
    print(f"  Train core:   up to {train_end.date()} ({train_core_mask.sum()} samples)")
    print(f"  Validation:   {val_start.date()} → {val_end.date()} ({val_mask.sum()} samples)")
    print(f"  Test:         {test_start.date()} → {test_end.date()} ({test_mask.sum()} samples)")

    return train_core_mask, val_mask, test_mask


def walk_forward_splits(
    dates: np.ndarray,
    val_months: int = WF_VAL_MONTHS,
    step_months: int = WF_STEP_MONTHS,
    min_train_months: int = WF_MIN_TRAIN_MONTHS,
    final_test_months: int = WF_FINAL_TEST_MONTHS,
):
    """
    Expanding-window walk-forward validation.

    Intended structure:
    Fold 1: Train 2000–2012 -> Validate 2013–2015
    Fold 2: Train 2000–2015 -> Validate 2016–2018
    Fold 3: Train 2000–2018 -> Validate 2019–2021
    Final test: 2022–2024 remains untouched
    """

    dates = pd.to_datetime(dates)

    if getattr(dates, "tz", None) is not None:
        dates = dates.tz_localize(None)

    last_date = dates.max()

    start_anchor = CUTOFF_DATE.tz_localize(None)

    final_test_start = (
        last_date
        - pd.DateOffset(months=final_test_months)
        + pd.Timedelta(days=1)
    )

    val_start = start_anchor + pd.DateOffset(months=min_train_months)

    splits = []
    fold = 1

    while True:
        val_end = val_start + pd.DateOffset(months=val_months) - pd.Timedelta(days=1)

        if val_end >= final_test_start:
            break

        train_end = val_start - pd.Timedelta(days=1)

        train_mask = dates <= train_end
        val_mask = (dates >= val_start) & (dates <= val_end)

        if train_mask.sum() > 0 and val_mask.sum() > 0:
            splits.append({
                "fold": fold,
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
                "train_mask": train_mask,
                "val_mask": val_mask,
            })

            print(
                f"WF fold {fold}: "
                f"train up to {train_end.date()} ({train_mask.sum()} samples), "
                f"validate {val_start.date()} → {val_end.date()} ({val_mask.sum()} samples)"
            )

        fold += 1
        val_start = val_start + pd.DateOffset(months=step_months)

    if not splits:
        raise ValueError("No walk-forward splits created. Check WF settings.")

    return splits


# -----------------------
# Cache helpers
# -----------------------

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
# Walk-forward hyperopt
# -----------------------

def hyperopt_single_scale_walk_forward(
    scale: str,
    lags_candidates,
    layer_configs,
    monthly_prices,
    weekly_prices,
    daily_prices,
):
    """
    Walk-forward hyperparameter optimization for one regression scale.
    Selects by stability-adjusted hit rate:

        stability_adjusted_hit_rate = mean_hit_rate - std_hit_rate
    """

    assert scale in {"monthly", "weekly", "daily"}

    best_config = None
    best_score = -np.inf
    best_summary = None

    all_results = []
    all_fold_details = []

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

        # Slice features for selected scale only
        if scale == "monthly":
            X_scale = X_tmp[:, :l_m]
        elif scale == "weekly":
            X_scale = X_tmp[:, l_m : l_m + l_w]
        else:
            X_scale = X_tmp[:, l_m + l_w :]

        wf_splits = walk_forward_splits(
            d_tmp,
            val_months=WF_VAL_MONTHS,
            step_months=WF_STEP_MONTHS,
            min_train_months=WF_MIN_TRAIN_MONTHS,
            final_test_months=WF_FINAL_TEST_MONTHS,
        )

        for layers in layer_configs:
            for solver in ["adam"]:

                lr_candidates = HYPEROPT_LEARNING_RATES if solver in ["adam", "sgd"] else [None]

                for lr_init in lr_candidates:
                    print(
                        f"\n--- {scale} | lags={lags_value} | layers={layers} | "
                        f"solver={solver} | lr_init={lr_init} | WALK-FORWARD ---"
                    )

                    threshold_fold_results = {
                        float(thr): []
                        for thr in HYPEROPT_THRESHOLDS
                    }

                    for split in wf_splits:
                        fold = split["fold"]
                        train_mask = split["train_mask"]
                        val_mask = split["val_mask"]

                        X_train = X_scale[train_mask]
                        X_val = X_scale[val_mask]

                        y_train = y_tmp[train_mask]
                        y_val = y_tmp[val_mask]

                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_val_scaled = scaler.transform(X_val)

                        common_params = dict(
                            hidden_layer_sizes=layers,
                            activation="relu",
                            solver=solver,
                            alpha=0.001,
                            max_iter=500,
                            random_state=42,
                            verbose=False,
                        )

                        if solver == "adam":
                            model = MLPRegressor(
                                **common_params,
                                learning_rate_init=lr_init,
                                early_stopping=True,
                                n_iter_no_change=20,
                                validation_fraction=0.10,
                            )
                        elif solver == "sgd":
                            model = MLPRegressor(
                                **common_params,
                                learning_rate_init=lr_init,
                                learning_rate="adaptive",
                                early_stopping=True,
                                n_iter_no_change=20,
                                validation_fraction=0.10,
                            )
                        else:
                            model = MLPRegressor(**common_params)

                        model.fit(X_train_scaled, y_train)

                        if (
                            hasattr(model, "loss_curve_")
                            and model.loss_curve_ is not None
                            and len(model.loss_curve_) > 0
                        ):
                            loss_curve_last = float(model.loss_curve_[-1])
                            loss_curve_len = len(model.loss_curve_)
                        else:
                            loss_curve_last = np.nan
                            loss_curve_len = 0

                        loss_final = float(model.loss_) if hasattr(model, "loss_") else np.nan

                        # Continuous predicted log returns
                        y_val_pred = model.predict(X_val_scaled)

                        baseline_hit_rate = float(np.mean(y_val >= 0))

                        for thr in HYPEROPT_THRESHOLDS:
                            thr = float(thr)

                            signal_mask = y_val_pred >= thr
                            coverage = float(signal_mask.mean())

                            if coverage == 0:
                                hit_rate = 0.0
                                mean_return = np.nan
                                n_signals = 0
                            else:
                                realized = y_val[signal_mask]
                                hit_rate = float(np.mean(realized >= 0))
                                mean_return = float(np.mean(realized))
                                n_signals = int(signal_mask.sum())

                            fold_row = {
                                "scale": scale,
                                "lags": lags_value,
                                "layers": str(layers),
                                "solver": solver,
                                "learning_rate_init": lr_init,
                                "threshold": thr,
                                "fold": fold,
                                "train_end": split["train_end"],
                                "val_start": split["val_start"],
                                "val_end": split["val_end"],
                                "hit_rate": hit_rate,
                                "coverage": coverage,
                                "n_signals": n_signals,
                                "mean_return": mean_return,
                                "baseline_hit_rate": baseline_hit_rate,
                                "hit_rate_minus_baseline": hit_rate - baseline_hit_rate,
                                "loss_curve_last": (
                                    float(loss_curve_last)
                                    if not np.isnan(loss_curve_last)
                                    else np.nan
                                ),
                                "loss_final": (
                                    float(loss_final)
                                    if not np.isnan(loss_final)
                                    else np.nan
                                ),
                                "loss_curve_len": int(loss_curve_len),
                            }

                            threshold_fold_results[thr].append(fold_row)
                            all_fold_details.append(fold_row)

                    # Summarize thresholds across walk-forward folds
                    for thr, fold_results in threshold_fold_results.items():
                        hit_rates = [r["hit_rate"] for r in fold_results]
                        coverages = [r["coverage"] for r in fold_results]
                        baselines = [r["baseline_hit_rate"] for r in fold_results]
                        mean_returns = [
                            r["mean_return"]
                            for r in fold_results
                            if not np.isnan(r["mean_return"])
                        ]

                        mean_hit_rate = float(np.mean(hit_rates))
                        std_hit_rate = (
                            float(np.std(hit_rates, ddof=1))
                            if len(hit_rates) > 1
                            else 0.0
                        )

                        mean_coverage = float(np.mean(coverages))
                        std_coverage = (
                            float(np.std(coverages, ddof=1))
                            if len(coverages) > 1
                            else 0.0
                        )

                        mean_baseline_hit_rate = float(np.mean(baselines))
                        hit_rate_minus_baseline = mean_hit_rate - mean_baseline_hit_rate

                        avg_mean_return = (
                            float(np.mean(mean_returns))
                            if len(mean_returns) > 0
                            else np.nan
                        )

                        stability_adjusted_hit_rate = mean_hit_rate - std_hit_rate

                        print(
                            f"    thr={thr:.4f} | "
                            f"mean_hit_rate={mean_hit_rate:.4f}, "
                            f"std_hit_rate={std_hit_rate:.4f}, "
                            f"stability_adjusted_hit_rate={stability_adjusted_hit_rate:.4f}, "
                            f"mean_coverage={mean_coverage:.4f}, "
                            f"mean_baseline={mean_baseline_hit_rate:.4f}, "
                            f"hit_rate_minus_baseline={hit_rate_minus_baseline:.4f}, "
                            f"mean_return={avg_mean_return:.5f}"
                        )

                        summary_row = {
                            "scale": scale,
                            "lags": lags_value,
                            "layers": str(layers),
                            "solver": solver,
                            "learning_rate_init": lr_init,
                            "threshold": float(thr),
                            "n_folds": len(fold_results),
                            "mean_hit_rate": mean_hit_rate,
                            "std_hit_rate": std_hit_rate,
                            "stability_adjusted_hit_rate": stability_adjusted_hit_rate,
                            "mean_coverage": mean_coverage,
                            "std_coverage": std_coverage,
                            "mean_baseline_hit_rate": mean_baseline_hit_rate,
                            "hit_rate_minus_baseline": hit_rate_minus_baseline,
                            "mean_return": avg_mean_return,
                        }

                        all_results.append(summary_row)

                        # Selection rule:
                        # Choose the model with the highest stability-adjusted hit rate,
                        # while requiring enough coverage and improvement over baseline.
                        if (
                            mean_coverage >= 0.05
                            and hit_rate_minus_baseline > 0
                            and stability_adjusted_hit_rate > best_score
                        ):
                            best_score = stability_adjusted_hit_rate

                            best_config = {
                                "scale": scale,
                                "lags": lags_value,
                                "layers": layers,
                                "solver": solver,
                                "learning_rate_init": lr_init,
                                "threshold": float(thr),
                            }

                            best_summary = {
                                "mean_hit_rate": mean_hit_rate,
                                "std_hit_rate": std_hit_rate,
                                "stability_adjusted_hit_rate": stability_adjusted_hit_rate,
                                "mean_coverage": mean_coverage,
                                "std_coverage": std_coverage,
                                "mean_baseline_hit_rate": mean_baseline_hit_rate,
                                "hit_rate_minus_baseline": hit_rate_minus_baseline,
                                "mean_return": avg_mean_return,
                                "n_folds": len(fold_results),
                            }

    # Save summary results
    if all_results:
        print("\n=== Walk-forward hyperopt summary sorted by stability_adjusted_hit_rate ===")

        all_results_sorted = sorted(
            all_results,
            key=lambda r: (r["stability_adjusted_hit_rate"], r["mean_coverage"]),
            reverse=True,
        )

        df_summary = pd.DataFrame(all_results_sorted)

        summary_csv = f"hyperopt_{scale}_regression_walk_forward_summary.csv"
        df_summary.to_csv(BASE_DIR / summary_csv, index=False)

        print(f"✅ Saved walk-forward summary results to {summary_csv}")

        print(
            df_summary[
                [
                    "scale",
                    "lags",
                    "layers",
                    "solver",
                    "learning_rate_init",
                    "threshold",
                    "n_folds",
                    "mean_hit_rate",
                    "std_hit_rate",
                    "stability_adjusted_hit_rate",
                    "mean_coverage",
                    "std_coverage",
                    "mean_baseline_hit_rate",
                    "hit_rate_minus_baseline",
                    "mean_return",
                ]
            ].head(30)
        )

    # Save fold-level results
    if all_fold_details:
        df_folds = pd.DataFrame(all_fold_details)

        fold_csv = f"hyperopt_{scale}_regression_walk_forward_fold_details.csv"
        df_folds.to_csv(BASE_DIR / fold_csv, index=False)

        print(f"✅ Saved fold-level results to {fold_csv}")

    print("\n=== Best hyperparameters for single-scale MLP regressor by stability-adjusted hit rate ===")
    print(best_config)
    print(best_summary)

    return best_config, best_summary


# -----------------------
# Main
# -----------------------

def main():
    t0 = time.time()

    # -----------------------
    # 1) Load raw price data
    # -----------------------

    wide = cache_load("wide")

    if wide is None:
        print("Loading daily wide prices...")

        if not DATA_FILE.exists():
            raise FileNotFoundError(
                f"Could not find data file: {DATA_FILE}\n"
                f"Make sure nasdaq_exchange_daily_price_data_close.csv "
                f"is in the same folder as this script."
            )

        wide = load_wide_prices(DATA_FILE)
        cache_save("wide", wide)
    else:
        print("Using cached wide data")

    # -----------------------
    # 2) Resample to monthly / weekly / daily
    # -----------------------

    monthly_prices = cache_load("monthly_prices")
    weekly_prices = cache_load("weekly_prices")
    daily_prices = cache_load("daily_prices")

    if monthly_prices is None or weekly_prices is None or daily_prices is None:
        print("Converting daily → monthly/weekly/daily...")

        monthly_prices = wide_to_prices(wide, freq="ME")
        weekly_prices = wide_to_prices(wide, freq="W")
        daily_prices = wide_to_prices(wide, freq="D")

        cache_save("monthly_prices", monthly_prices)
        cache_save("weekly_prices", weekly_prices)
        cache_save("daily_prices", daily_prices)
    else:
        print("Using cached monthly/weekly/daily")

    # -----------------------
    # 3) Build default supervised arrays for sample-size check
    # -----------------------

    supervised = cache_load("supervised_arrays_regression")

    if supervised is None:
        print("Building default supervised dataset...")

        X, y, d, stocks = build_supervised_multiscale(
            monthly=monthly_prices,
            weekly=weekly_prices,
            daily=daily_prices,
            lags_m=HISTORY_LAGS_M,
            lags_w=HISTORY_LAGS_W,
            lags_d=HISTORY_LAGS_D,
        )

        cache_save("supervised_arrays_regression", (X, y, d, stocks))
        joblib.dump(stocks, MODEL_DIR / "stocks.joblib")
    else:
        print("Using cached supervised arrays")
        X, y, d, stocks = supervised

    print(
        f"✅ Samples: {X.shape[0]:,} | Features: {X.shape[1]} "
        f"(M:{HISTORY_LAGS_M} + W:{HISTORY_LAGS_W} + D:{HISTORY_LAGS_D}) "
        f"| Mean target return: {y.mean():.5f} | Std: {y.std():.5f}"
    )

    # -----------------------
    # 4) Run walk-forward hyperparameter search
    # -----------------------

    if RUN_SINGLE_SCALE_HYPEROPT and USE_WALK_FORWARD:
        print("Start walk-forward hyperopt")

        hyperopt_single_scale_walk_forward(
            scale=HYPEROPT_SCALE,
            lags_candidates=HYPEROPT_LAGS,
            layer_configs=HYPEROPT_LAYER_CONFIGS,
            monthly_prices=monthly_prices,
            weekly_prices=weekly_prices,
            daily_prices=daily_prices,
        )

        print("End walk-forward hyperopt")

        elapsed = time.time() - t0

        print(
            f"\n⏱️ Total runtime with walk-forward hyperopt: "
            f"{elapsed / 60:.2f} minutes ({elapsed:.1f} seconds)"
        )

        return

    print("Set RUN_SINGLE_SCALE_HYPEROPT=True and USE_WALK_FORWARD=True to run walk-forward optimisation.")


if __name__ == "__main__":
    main()