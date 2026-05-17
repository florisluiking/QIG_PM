"""
backtest_walkforward.py
────────────────────────
Walk-forward backtest with incremental model updating via partial_fit().

Loop for every test month T:
  1. Predict on month T  (model has NOT seen T yet)
  2. Collect portfolio returns for month T
  3. partial_fit() all 6 models on month T's data   ← the incremental update
  4. Move to month T+1

Everything else (loading, scoring, ranking, plotting) is identical to
backtest3_6_6.py so results are directly comparable.

Toggle VAL_WARMUP = True to first warm up on the full val period before
the walk-forward test loop begins.
"""

import copy
import itertools
import math
import statistics
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import yfinance as yf
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

# ── config (keep in sync with your training script) ──────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
CACHE_DIR = Path("cache_simple")

MODEL_LAGS = {
    "monthly":   50,
    "weekly":   100,
    "daily":    100,
    "monthly_p": 50,
    "weekly_p": 150,
    "daily_p":   50,
}

MODEL_FEATURE_BLOCK = {
    "monthly":   "monthly",
    "weekly":    "weekly",
    "daily":     "daily",
    "monthly_p": "monthly",
    "weekly_p":  "weekly",
    "daily_p":   "daily",
}

MODEL_THRESHOLD_GRIDS = {
    "monthly":   [0.52],
    "weekly":    [0.52],
    "daily":     [0.52],
    "monthly_p": [-0.004],
    "weekly_p":  [-0.004],
    "daily_p":   [-0.004],
}

CLASSIFICATION_MODELS = ("monthly", "weekly", "daily")
PREDICTION_MODELS     = ("monthly_p", "weekly_p", "daily_p")

VAL_MONTHS  = 36
TEST_MONTHS = 36

# ── walk-forward knob ─────────────────────────────────────────────────────────
VAL_WARMUP = False   # set True to partial_fit on entire val period before test


# ── helpers ───────────────────────────────────────────────────────────────────
def cache_load(name):
    path = CACHE_DIR / f"{name}.joblib"
    if path.exists():
        print(f"🔄  Loading cached {name}")
        return joblib.load(path)
    return None


def split_masks(dates, val_months=VAL_MONTHS, test_months=TEST_MONTHS):
    dates = pd.to_datetime(dates)
    if getattr(dates, "tz", None) is not None:
        dates = dates.tz_localize(None)
    last_date  = dates.max()
    train_end  = last_date - pd.DateOffset(months=(val_months + test_months))
    val_start  = train_end  + pd.DateOffset(months=1)
    val_end    = last_date  - pd.DateOffset(months=test_months)
    test_start = val_end    + pd.DateOffset(months=1)
    train_mask = dates <= train_end
    val_mask   = (dates >= val_start)  & (dates <= val_end)
    test_mask  = (dates >= test_start) & (dates <= last_date)
    print("Dynamic split dates:")
    print(f"  Train: up to {train_end.date()} ({train_mask.sum()} samples)")
    print(f"  Val:   {val_start.date()} → {val_end.date()} ({val_mask.sum()} samples)")
    print(f"  Test:  {test_start.date()} → {last_date.date()} ({test_mask.sum()} samples)")
    return train_mask, val_mask, test_mask


def load_model(path):
    bundle = joblib.load(path)
    path_obj = Path(path)
    if "sklearn_version" in bundle:
        if bundle["sklearn_version"] != sklearn.__version__:
            raise RuntimeError(
                f"{path_obj.name} trained with sklearn {bundle['sklearn_version']}, "
                f"current={sklearn.__version__}"
            )
    else:
        print(f"⚠️  No sklearn version in {path_obj.name}")
    return bundle


def score_block(bundle, X_full, mask, block_offsets, model_name):
    """Return model scores (proba or predicted return) for rows in `mask`."""
    block   = MODEL_FEATURE_BLOCK[model_name]
    start   = block_offsets[block]
    lags    = MODEL_LAGS[model_name]
    X_slice = X_full[mask, start:start + lags]
    X_sc    = bundle["scaler"].transform(X_slice)
    if model_name in CLASSIFICATION_MODELS:
        return bundle["model"].predict_proba(X_sc)[:, 1]
    else:
        return bundle["model"].predict(X_sc)


# ═════════════════════════════════════════════════════════════════════════════
def main():
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    t0 = time.time()

    # ── 1. Load models ────────────────────────────────────────────────────────
    model_paths = {
        "monthly": BASE_DIR / "artifacts" / "monthly_classification_lags50_layers_8_4_sgd_lr00005.joblib",
        "weekly": BASE_DIR / "artifacts" / "weekly_classification_lags100_layers8_4_sgd_lr00005.joblib",
        "daily": BASE_DIR / "artifacts" / "daily_classification_lags100_layers_128_64_adam_lr0005.joblib",
        "monthly_p": BASE_DIR / "artifacts" / "monthly_regression_lags50_layers8_4_sgd_lr0001.joblib",
        "weekly_p": BASE_DIR / "artifacts" / "weekly_regression_lags150_layers8_4_lbfgs.joblib",
        "daily_p": BASE_DIR / "artifacts" / "daily_regression_lags50_layers32_16_8_adam_lr0005.joblib",
    }

    loaded_models = {}
    for name, path in model_paths.items():
        print(f"Loading {name} …")
        loaded_models[name] = load_model(path)

    # ── 2. Load supervised cache ───────────────────────────────────────────────
    supervised = cache_load("supervised_arrays_class")
    X, y, d, R, stocks = supervised

    train_mask, val_mask, test_mask = split_masks(d)
    
    
    # Disable early_stopping and reset stale training state for partial_fit
    for name, bundle in loaded_models.items():
        bundle["model"].early_stopping = False
        bundle["model"].best_loss_ = np.inf
        bundle["model"]._no_improvement_count = 0

    # ── 3. Feature-block offsets (hardcoded to match training layout) ───────────
    # X is 300 wide: monthly[0:50], weekly[50:200], daily[200:300]
    block_offsets = {
        "monthly": 0,
        "weekly":  50,
        "daily":   200,
    }

    dates_all = pd.to_datetime(d)
    if getattr(dates_all, "tz", None) is not None:
        dates_all = dates_all.tz_localize(None)

    stocks_arr = stocks  # already unpacked from supervised tuple

    # ── 4. Optional: val warm-up ──────────────────────────────────────────────
    # partial_fit on every val month in chronological order so the models have
    # seen recent history before the test period begins.
    if VAL_WARMUP:
        print("\n🔥 Val warm-up: partial_fit on each val month …")
        val_months_sorted = sorted(pd.to_datetime(d[val_mask]).unique())
        for vm in val_months_sorted:
            vm_mask = (dates_all == vm) & val_mask
            if vm_mask.sum() == 0:
                continue
            y_vm = y[vm_mask]
            for model_name, bundle in loaded_models.items():
                block  = MODEL_FEATURE_BLOCK[model_name]
                start  = block_offsets[block]
                lags   = MODEL_LAGS[model_name]
                X_sl   = X[vm_mask, start:start + lags]
                X_sc   = bundle["scaler"].transform(X_sl)
                # ── partial_fit (warm-up) ────────────────────────────────────
                if model_name in CLASSIFICATION_MODELS:
                    bundle["model"].partial_fit(X_sc, y_vm, classes=np.array([0, 1]))
                else:
                    bundle["model"].partial_fit(X_sc, y_vm)
        print("   Warm-up complete.")

    # ── 5. Walk-forward test loop ─────────────────────────────────────────────
    #
    #   For each test month T (chronological order):
    #     a) Score all stocks in T  → collect signals
    #     b) Record portfolio returns for T
    #     c) *** partial_fit all 6 models on month T's data ***   ← HERE
    #     d) Advance to T+1
    #
    print("\n🚀 Starting walk-forward backtest …")

    _test_dates = pd.to_datetime(d[test_mask])
    if getattr(_test_dates, 'tz', None) is not None:
        _test_dates = _test_dates.tz_localize(None)
    test_months_sorted = sorted(_test_dates.unique())

    # Storage for walk-forward results
    monthly_rankings = {}   # month → ordered list of stocks
    R_test_dict      = {}   # month → dict {stock: realized_return}

    # For compatibility with the threshold sweep we collect all scores first
    # (they are produced one month at a time but stored together for the sweep).
    all_scores   = {m: [] for m in loaded_models}
    all_y_test   = []
    all_R_test   = []
    all_dates_wf = []
    all_stocks_wf= []

    for T_idx, month in enumerate(test_months_sorted):
        month_mask = (dates_all == month) & test_mask

        if month_mask.sum() == 0:
            continue

        # ── a) Score this month ───────────────────────────────────────────────
        month_scores = {}
        for model_name, bundle in loaded_models.items():
            month_scores[model_name] = score_block(
                bundle, X, month_mask, block_offsets, model_name
            )

        # Store for later threshold sweep + ranking
        for model_name in loaded_models:
            all_scores[model_name].append(month_scores[model_name])
        all_y_test.append(y[month_mask])
        all_R_test.append(R[month_mask])
        all_dates_wf.append(dates_all[month_mask])
        all_stocks_wf.append(stocks_arr[month_mask])

        # ── c) *** INCREMENTAL UPDATE — partial_fit on month T *** ────────────
        #
        #   We now expose month T's true labels to the model so it can adapt.
        #   This is the core of the walk-forward update.
        #   Classification models need `classes` on every partial_fit call.
        #
        y_month = y[month_mask]
        for model_name, bundle in loaded_models.items():
            block  = MODEL_FEATURE_BLOCK[model_name]
            start  = block_offsets[block]
            lags   = MODEL_LAGS[model_name]
            X_sl   = X[month_mask, start:start + lags]
            X_sc   = bundle["scaler"].transform(X_sl)

            # ── partial_fit HERE ─────────────────────────────────────────────
            if model_name in CLASSIFICATION_MODELS:
                bundle["model"].partial_fit(X_sc, y_month, classes=np.array([0, 1]))
            else:
                bundle["model"].partial_fit(X_sc, y_month)
            # ── end partial_fit ──────────────────────────────────────────────

        print(f"  ✅ Month {month.date() if hasattr(month,'date') else month}  "
              f"| samples: {month_mask.sum():4d}  "
              f"| cumulative months done: {T_idx+1}/{len(test_months_sorted)}")

    # ── 6. Concatenate walk-forward results ───────────────────────────────────
    wf_scores   = {m: np.concatenate(all_scores[m]) for m in loaded_models}
    y_test_wf   = np.concatenate(all_y_test)
    R_test_wf   = np.concatenate(all_R_test)
    dates_wf    = np.concatenate(all_dates_wf)
    stocks_wf   = np.concatenate(all_stocks_wf)

    p_m, p_w, p_d     = wf_scores["monthly"], wf_scores["weekly"], wf_scores["daily"]
    X_mp, X_wp, X_dp  = wf_scores["monthly_p"], wf_scores["weekly_p"], wf_scores["daily_p"]

    # ── 7. Threshold sweep (identical to original) ────────────────────────────
    ordered_models          = list(MODEL_THRESHOLD_GRIDS.keys())
    threshold_value_lists   = [MODEL_THRESHOLD_GRIDS[m] for m in ordered_models]

    results = []
    for combo in itertools.product(*threshold_value_lists):
        threshold_map   = dict(zip(ordered_models, combo))
        signal_mask     = np.logical_and.reduce(
            [wf_scores[m] >= threshold_map[m] for m in ordered_models]
        )
        n_signals = signal_mask.sum()
        if n_signals == 0:
            continue
        realized_returns = R_test_wf[signal_mask]
        binary_preds     = {m: (wf_scores[m] >= threshold_map[m]).astype(int) for m in ordered_models}
        y_pred_combined  = np.logical_and.reduce(
            [binary_preds[m] == 1 for m in ordered_models]
        ).astype(int)

        from sklearn.metrics import classification_report
        print(f"\n=== RESULTS thresholds {threshold_map} ===")
        print(classification_report(y_test_wf, y_pred_combined, digits=4))
        agreement = np.mean(y_pred_combined == y_test_wf)
        print(f"✅ Agreement: {agreement*100:.2f}%")

        row = {
            "coverage":        n_signals / len(y_test_wf),
            "n_signals":       n_signals,
            "hit_rate":        np.mean(realized_returns > 0),
            "mean_return":     np.mean(realized_returns),
            "median_return":   np.median(realized_returns),
            "variance":        np.var(realized_returns),
            "min_return":      np.min(realized_returns),
            "max_return":      np.max(realized_returns),
            "quantile_5":      np.percentile(realized_returns, 5),
            "quantile_10":     np.percentile(realized_returns, 10),
            "quantile_90":     np.percentile(realized_returns, 90),
            "quantile_95":     np.percentile(realized_returns, 95),
        }
        for m in ordered_models:
            row[f"thr_{m}"] = threshold_map[m]
        results.append(row)

    df_results = pd.DataFrame(results).sort_values("mean_return", ascending=False).reset_index(drop=True)
    df_results.to_csv("df_results_walkforward.csv", index=False)
    print("\n=== Threshold results ===")
    print(df_results)

    best_thresholds = {m: df_results.iloc[0][f"thr_{m}"] for m in ordered_models}
    print(f"\nBest thresholds: {best_thresholds}")

    thr_monthly   = best_thresholds["monthly"]
    thr_weekly    = best_thresholds["weekly"]
    thr_daily     = best_thresholds["daily"]
    thr_monthly_p = best_thresholds["monthly_p"]
    thr_weekly_p  = best_thresholds["weekly_p"]
    thr_daily_p   = best_thresholds["daily_p"]

    # ── 8. Monthly rankings (identical logic to original) ─────────────────────
    dates_test_pd = pd.to_datetime(dates_wf)

    var_ex   = statistics.variance(X_mp)
    var_prob = statistics.variance(p_m)
    factor   = np.sqrt(var_prob) / np.sqrt(var_ex)
    w_ret, w_proba = 1.0, 1.0

    monthly_rankings = {}
    for month in sorted(dates_test_pd.unique()):
        idx = dates_test_pd == month
        if idx.sum() == 0:
            continue
        mask = (
            (p_m[idx]  >= thr_monthly)   &
            (p_w[idx]  >= thr_weekly)    &
            (p_d[idx]  >= thr_daily)     &
            (X_mp[idx] >= thr_monthly_p) &
            (X_wp[idx] >= thr_weekly_p)  &
            (X_dp[idx] >= thr_daily_p)
        )
        if mask.sum() == 0:
            continue
        avg_ret   = (X_mp[idx][mask] + X_wp[idx][mask] + X_dp[idx][mask]) / 3.0
        avg_proba = (p_m[idx][mask]  + p_w[idx][mask]  + p_d[idx][mask])  / 3.0
        avg_pred_thr  = np.mean([thr_monthly_p, thr_weekly_p, thr_daily_p])
        avg_class_thr = np.mean([thr_monthly, thr_weekly, thr_daily])
        score = w_ret * (avg_ret - avg_pred_thr) + w_proba * ((avg_proba - avg_class_thr) / factor)
        df_month = pd.DataFrame({
            "stock": stocks_wf[idx][mask],
            "score": score,
        }).sort_values("score", ascending=False)
        monthly_rankings[month] = df_month["stock"].tolist()

    # ── 9. Portfolio simulation (identical to original) ───────────────────────
    months          = sorted(monthly_rankings.keys())
    portfolio5      = [2000]
    portfolio10     = [2000]
    portfolio1      = [2000]
    returns_5, returns_10, ret_1 = [], [], []
    through_mask_n  = []

    for i, m in enumerate(months, start=1):
        through_mask_n.append(len(monthly_rankings[m]))
        ranked = monthly_rankings[m]
        picks5  = ranked[:5]  if len(ranked) >= 5  else ranked
        picks10 = ranked[:10] if len(ranked) >= 10 else ranked
        pick1   = ranked[0]

        def get_return(stock, month):
            mask = (stocks_wf == stock) & (dates_test_pd == month)
            return R_test_wf[mask][0] if np.any(mask) else 0.0

        # Top-5
        win5 = sum(
            (portfolio5[i-1] / len(picks5)) * (math.exp(get_return(s, m)) - 1)
            for s in picks5
        )
        returns_5.append(win5 / portfolio5[i-1])
        portfolio5.append(portfolio5[i-1] + win5)

        # Top-10
        win10 = sum(
            (portfolio10[i-1] / len(picks10)) * (math.exp(get_return(s, m)) - 1)
            for s in picks10
        )
        returns_10.append(win10 / portfolio10[i-1])
        portfolio10.append(portfolio10[i-1] + win10)

        # Top-1
        r1 = get_return(pick1, m)
        ret_1.append(r1)
        portfolio1.append(portfolio1[i-1] * math.exp(r1))

    # ── 10. Metrics ───────────────────────────────────────────────────────────
    r_f = 0.0191
    for label, rets in [("5-stock", returns_5), ("10-stock", returns_10), ("1-stock", ret_1)]:
        E_R    = 12 * np.mean(rets)
        sig    = np.sqrt(12) * np.std(rets, ddof=1)
        sharpe = (E_R - r_f) / sig
        maxdd  = np.min(rets)
        print(f"\n{label}:  Sharpe={sharpe:.3f}  MaxDrawdown={maxdd:.4f}")

    # S&P 500 benchmark
    ticker  = yf.Ticker("^GSPC")
    sp500   = ticker.history(start="2022-01-01", end="2025-01-01", auto_adjust=True)["Close"]
    sp500.index = pd.to_datetime(sp500.index).tz_localize(None)
    sp500   = sp500.sort_index()
    sp500   = sp500[~sp500.index.duplicated(keep="last")]

    initial_month   = pd.Timestamp("2022-08-30")
    months_with_ini = [initial_month] + [pd.Timestamp(m).tz_localize(None) if hasattr(m, 'tz') else pd.Timestamp(m) for m in months]
    months_clean    = pd.to_datetime(months_with_ini).to_period("M").to_timestamp()
    sp_aligned      = sp500.reindex(months_clean, method="ffill")
    sp_rets         = sp_aligned.pct_change().fillna(0).values
    sharpe_sp       = (12 * np.mean(sp_rets) - r_f) / (np.sqrt(12) * np.std(sp_rets, ddof=1))
    print(f"\nS&P 500:  Sharpe={sharpe_sp:.3f}  MaxDrawdown={np.min(sp_rets):.4f}")

    # ── 11. Plot ───────────────────────────────────────────────────────────────
    cum5   = np.insert(np.cumprod(1 + np.array(returns_5)),   0, 1.0)
    cum10  = np.insert(np.cumprod(1 + np.array(returns_10)),  0, 1.0)
    cum1   = np.insert(np.cumprod(1 + np.array(ret_1)),       0, 1.0)
    cum_sp = np.cumprod(1 + sp_rets)

    plt.figure(figsize=(12, 6))
    plt.plot(months_clean, cum1,   label="Top 1  (retraining)")
    plt.plot(months_clean, cum5,   label="Top 5  (retraining)")
    plt.plot(months_clean, cum10,  label="Top 10 (retraining)")
    plt.plot(months_clean, cum_sp, label="S&P 500")
    plt.yscale("log")
    plt.title("Retraining Backtest — Cumulative Returns (with partial_fit updates)")
    plt.xlabel("Month")
    plt.ylabel("Cumulative Return (log scale)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("walkforward_cumulative_returns.png", dpi=150)
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(through_mask_n)
    plt.title("Stocks passing all 6 filters per month (walk-forward)")
    plt.xlabel("Month index")
    plt.ylabel("# stocks")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"\n⏱️  Total runtime: {(time.time()-t0)/60:.2f} min")


if __name__ == "__main__":
    main()
