import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score
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

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "nasdaq_exchange_daily_price_data_close.csv"

CACHE_DIR = BASE_DIR / "cache_simple"
CACHE_DIR.mkdir(exist_ok=True)

MODEL_DIR = BASE_DIR / "artifacts"
MODEL_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# Config
# --------------------------------------------------

CUTOFF_DATE = pd.Timestamp("2000-01-01", tz="UTC")

HISTORY_LAGS_M = 100
HISTORY_LAGS_W = 100
HISTORY_LAGS_D = 100

# Current normal final test setup
VAL_MONTHS = 36
TEST_MONTHS = 36

# Hyperparameter search config
RUN_SINGLE_SCALE_HYPEROPT = True

# Change this to "monthly", "weekly", or "daily"
HYPEROPT_SCALE = "daily"

HYPEROPT_LAGS = [50, 100, 150]

HYPEROPT_LAYER_CONFIGS = [
    (128, 64),
    (32, 16, 8),
    (8, 4),
]

HYPEROPT_LEARNING_RATES = [0.0001, 0.0005, 0.001, 0.005]

# --------------------------------------------------
# Walk-forward validation settings: Option A
# --------------------------------------------------

USE_WALK_FORWARD = True

WF_VAL_MONTHS = 36          # each validation window = 3 years
WF_STEP_MONTHS = 36         # move forward by 3 years
WF_MIN_TRAIN_MONTHS = 156   # first training period ≈ 2000–2012
WF_FINAL_TEST_MONTHS = 36   # final 3 years are untouched for final testing

print("Made it")


# --------------------------------------------------
# Loading and preprocessing
# --------------------------------------------------

def find_first_datetime_row(csv_path: str) -> int:
    """Find first row whose first column parses as datetime."""
    raw = pd.read_csv(csv_path, header=None, nrows=500, low_memory=False)

    for idx, val in enumerate(raw.iloc[:, 0]):
        try:
            pd.to_datetime(val)
            return idx
        except Exception:
            continue

    return 0


def load_wide_prices(csv_path: str) -> pd.DataFrame:
    """Load wide daily prices, clean, and cut from CUTOFF_DATE."""
    first_data_row = find_first_datetime_row(csv_path)

    df = pd.read_csv(csv_path, skiprows=range(first_data_row), low_memory=False)

    df.rename(columns={df.columns[0]: "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"])

    df = df[df["date"] >= CUTOFF_DATE].reset_index(drop=True)

    return df


def wide_to_prices(df_wide: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    """
    Convert wide daily stock prices into long-format prices
    with user-specified frequency: daily, weekly, or monthly.
    """

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
    Build X, y, dates, realized returns, and stock labels.

    Features:
    - monthly log-return window
    - weekly log-return window
    - daily log-return window

    Label:
    - 1 if next month's price is higher than current month's price
    - 0 otherwise
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
    Real_list = []
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
        m_prices = gm["price"].to_numpy()
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

            # Label: next-month direction
            label = int(m_prices[t + 1] > m_prices[t])

            realized_log_return = m_ret[t + 1]

            if np.isnan(realized_log_return):
                continue

            x = np.concatenate([m_window, w_window, d_window], dtype=np.float32)

            X_list.append(x.astype(np.float32))
            Y_list.append(label)
            Real_list.append(realized_log_return)
            D_list.append(t_date)
            Stock_list.append(s)

    if not X_list:
        raise ValueError("No samples built. Check lags and data availability.")

    X = np.vstack(X_list).astype(np.float32)
    y = np.asarray(Y_list, dtype=np.int32)
    d_out = np.asarray(D_list)
    R = np.asarray(Real_list, dtype=np.float32)
    stocks_out = np.asarray(Stock_list, dtype=object)

    return X, y, d_out, R, stocks_out


# --------------------------------------------------
# Split functions
# --------------------------------------------------

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

    Option A intended structure:
    - Train 2000–2012 → Validate 2013–2015
    - Train 2000–2015 → Validate 2016–2018
    - Train 2000–2018 → Validate 2019–2021
    - Final test 2022–2024 remains untouched

    Actual masks use available supervised observations, so early dates may start later
    if lag requirements remove early observations.
    """

    dates = pd.to_datetime(dates)

    if getattr(dates, "tz", None) is not None:
        dates = dates.tz_localize(None)

    last_date = dates.max()

    # Use the raw-data cutoff as the conceptual start date
    start_anchor = CUTOFF_DATE.tz_localize(None)

    # Final test period starts after the last final_test_months
    final_test_start = last_date - pd.DateOffset(months=final_test_months) + pd.Timedelta(days=1)

    # First validation starts after the minimum training period
    val_start = start_anchor + pd.DateOffset(months=min_train_months)

    splits = []
    fold = 1

    while True:
        val_end = val_start + pd.DateOffset(months=val_months) - pd.Timedelta(days=1)

        # Do not let validation overlap with final test period
        if val_end >= final_test_start:
            break

        train_end = val_start - pd.Timedelta(days=1)

        train_mask = dates <= train_end
        val_mask = (dates >= val_start) & (dates <= val_end)

        if train_mask.sum() > 0 and val_mask.sum() > 0:
            splits.append({
                "fold": fold,
                "train_start": start_anchor,
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


# --------------------------------------------------
# Cache helpers
# --------------------------------------------------

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


# --------------------------------------------------
# Hyperparameter optimization
# --------------------------------------------------

def hyperopt_single_scale_walk_forward(
    scale: str,
    lags_candidates,
    layer_configs,
    monthly_prices,
    weekly_prices,
    daily_prices,
):
    """
    Walk-forward hyperparameter optimization for one scale:
    monthly, weekly, or daily classification.
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

        X_tmp, y_tmp, d_tmp, R_tmp, stocks_tmp = build_supervised_multiscale(
            monthly=monthly_prices,
            weekly=weekly_prices,
            daily=daily_prices,
            lags_m=l_m,
            lags_w=l_w,
            lags_d=l_d,
        )

        # Slice out only the selected scale's features
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
                lr_candidates = HYPEROPT_LEARNING_RATES

                for lr_init in lr_candidates:
                    print(
                        f"\n--- {scale} | lags={lags_value} | layers={layers} | "
                        f"solver={solver} | lr_init={lr_init} | WALK-FORWARD ---"
                    )

                    thresholds = np.arange(0.40, 0.60 + 1e-9, 0.02)
                    threshold_fold_results = {float(thr): [] for thr in thresholds}

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

                        model = MLPClassifier(
                            hidden_layer_sizes=layers,
                            activation="relu",
                            solver=solver,
                            alpha=0.001,
                            max_iter=500,
                            random_state=42,
                            verbose=False,
                            learning_rate_init=lr_init,
                            early_stopping=True,
                            n_iter_no_change=20,
                            validation_fraction=0.10,
                        )

                        model.fit(X_train_scaled, y_train)

                        y_val_prob = model.predict_proba(X_val_scaled)[:, 1]

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

                        for thr in thresholds:
                            thr = float(thr)

                            y_val_pred = (y_val_prob >= thr).astype(int)

                            coverage = (y_val_pred == 1).mean()
                            precision_1 = precision_score(
                                y_val,
                                y_val_pred,
                                pos_label=1,
                                zero_division=0,
                            )

                            baseline_positive_rate = y_val.mean()

                            fold_result = {
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
                                "precision_1": float(precision_1),
                                "coverage": float(coverage),
                                "baseline_positive_rate": float(baseline_positive_rate),
                                "precision_minus_baseline": float(precision_1 - baseline_positive_rate),
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

                            threshold_fold_results[thr].append(fold_result)
                            all_fold_details.append(fold_result)

                    # Summarize across folds for each threshold
                    for thr, fold_results in threshold_fold_results.items():
                        precisions = [r["precision_1"] for r in fold_results]
                        coverages = [r["coverage"] for r in fold_results]
                        baselines = [r["baseline_positive_rate"] for r in fold_results]

                        mean_precision = float(np.mean(precisions))
                        std_precision = (
                            float(np.std(precisions, ddof=1))
                            if len(precisions) > 1
                            else 0.0
                        )

                        mean_coverage = float(np.mean(coverages))
                        std_coverage = (
                            float(np.std(coverages, ddof=1))
                            if len(coverages) > 1
                            else 0.0
                        )

                        mean_baseline = float(np.mean(baselines))
                        precision_minus_baseline = mean_precision - mean_baseline

                        print(
                            f"    thr={thr:.2f} | "
                            f"mean_precision={mean_precision:.4f}, "
                            f"std_precision={std_precision:.4f}, "
                            f"mean_coverage={mean_coverage:.4f}, "
                            f"mean_baseline={mean_baseline:.4f}, "
                            f"precision_minus_baseline={precision_minus_baseline:.4f}"
                        )

                        summary_row = {
                            "scale": scale,
                            "lags": lags_value,
                            "layers": str(layers),
                            "solver": solver,
                            "learning_rate_init": lr_init,
                            "threshold": float(thr),
                            "n_folds": len(fold_results),
                            "mean_precision_1": mean_precision,
                            "std_precision_1": std_precision,
                            "mean_coverage": mean_coverage,
                            "std_coverage": std_coverage,
                            "mean_baseline_positive_rate": mean_baseline,
                            "precision_minus_baseline": precision_minus_baseline,
                        }

                        all_results.append(summary_row)

                        # Selection rule:
                        # highest average precision, with minimum average coverage
                        if mean_coverage >= 0.05 and mean_precision > best_score:
                            best_score = mean_precision

                            best_config = {
                                "scale": scale,
                                "lags": lags_value,
                                "layers": layers,
                                "solver": solver,
                                "learning_rate_init": lr_init,
                                "threshold": float(thr),
                            }

                            best_summary = {
                                "mean_precision_1": mean_precision,
                                "std_precision_1": std_precision,
                                "mean_coverage": mean_coverage,
                                "std_coverage": std_coverage,
                                "mean_baseline_positive_rate": mean_baseline,
                                "precision_minus_baseline": precision_minus_baseline,
                                "n_folds": len(fold_results),
                            }

    # Save summary results
    if all_results:
        all_results_sorted = sorted(
            all_results,
            key=lambda r: (r["mean_precision_1"], r["mean_coverage"]),
            reverse=True,
        )

        df_summary = pd.DataFrame(all_results_sorted)

        summary_csv = f"hyperopt_{scale}_classification_walk_forward_summary.csv"
        df_summary.to_csv(BASE_DIR / summary_csv, index=False)

        print(f"\n✅ Saved walk-forward summary results to {summary_csv}")

        print("\n=== Walk-forward hyperopt summary, sorted by mean precision ===")
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
                    "mean_precision_1",
                    "std_precision_1",
                    "mean_coverage",
                    "std_coverage",
                    "mean_baseline_positive_rate",
                    "precision_minus_baseline",
                ]
            ].head(30)
        )

    # Save fold-level details
    if all_fold_details:
        df_folds = pd.DataFrame(all_fold_details)

        fold_csv = f"hyperopt_{scale}_classification_walk_forward_fold_details.csv"
        df_folds.to_csv(BASE_DIR / fold_csv, index=False)

        print(f"✅ Saved fold-level results to {fold_csv}")

    print("\n=== Best hyperparameters by walk-forward mean precision with mean coverage >= 0.05 ===")
    print(best_config)
    print(best_summary)

    return best_config, best_summary


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    t0 = time.time()

    wide = cache_load("wide")

    if wide is None:
        print("Loading daily wide prices...")

        if not DATA_FILE.exists():
            raise FileNotFoundError(
                f"Could not find data file: {DATA_FILE}\n"
                f"Make sure nasdaq_exchange_daily_price_data_close.csv is in the same folder as this script."
            )

        wide = load_wide_prices(DATA_FILE)
        cache_save("wide", wide)
    else:
        print("Using cached wide data")

    monthly_prices = cache_load("monthly_prices")
    weekly_prices = cache_load("weekly_prices")
    daily_prices = cache_load("daily_prices")

    if monthly_prices is None or weekly_prices is None or daily_prices is None:
        print("Converting daily data to monthly, weekly, and daily formats...")

        monthly_prices = wide_to_prices(wide, freq="ME")
        weekly_prices = wide_to_prices(wide, freq="W")
        daily_prices = wide_to_prices(wide, freq="D")

        cache_save("monthly_prices", monthly_prices)
        cache_save("weekly_prices", weekly_prices)
        cache_save("daily_prices", daily_prices)
    else:
        print("Using cached monthly/weekly/daily prices")

    # Optional default supervised array cache, mainly for checking sample size
    supervised = cache_load("supervised_arrays_class")

    if supervised is None:
        print("Building default supervised dataset...")

        X, y, d, R, stocks = build_supervised_multiscale(
            monthly=monthly_prices,
            weekly=weekly_prices,
            daily=daily_prices,
            lags_m=HISTORY_LAGS_M,
            lags_w=HISTORY_LAGS_W,
            lags_d=HISTORY_LAGS_D,
        )

        cache_save("supervised_arrays_class", (X, y, d, R, stocks))
        joblib.dump(stocks, MODEL_DIR / "stocks.joblib")
    else:
        print("Using cached default supervised arrays")
        X, y, d, R, stocks = supervised

    print(
        f"✅ Default samples: {X.shape[0]:,} | Features: {X.shape[1]} "
        f"(M:{HISTORY_LAGS_M} + W:{HISTORY_LAGS_W} + D:{HISTORY_LAGS_D}) "
        f"| Positives: {y.sum():,} ({y.mean() * 100:.2f}%)"
    )

    if RUN_SINGLE_SCALE_HYPEROPT and USE_WALK_FORWARD:
        hyperopt_single_scale_walk_forward(
            scale=HYPEROPT_SCALE,
            lags_candidates=HYPEROPT_LAGS,
            layer_configs=HYPEROPT_LAYER_CONFIGS,
            monthly_prices=monthly_prices,
            weekly_prices=weekly_prices,
            daily_prices=daily_prices,
        )

        elapsed = time.time() - t0

        print(
            f"\n⏱️ Total runtime with walk-forward hyperopt: "
            f"{elapsed / 60:.2f} minutes ({elapsed:.1f} seconds)"
        )

        return

    print("Set RUN_SINGLE_SCALE_HYPEROPT=True and USE_WALK_FORWARD=True to run walk-forward optimization.")


if __name__ == "__main__":
    main()