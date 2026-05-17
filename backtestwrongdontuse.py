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
from sklearn.metrics import classification_report, f1_score, precision_score
from sklearn.exceptions import InconsistentVersionWarning
import joblib
import warnings
import json
import math
import itertools
import matplotlib.pyplot as plt
import yfinance as yf
import statistics
from pathlib import Path
import warnings
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = Path("cache_simple")
CACHE_DIR.mkdir(exist_ok=True)
# Per-model lag selection (can differ within same horizon)
MODEL_LAGS = {
    "monthly": 50,
    "weekly": 100,
    "daily": 100,
    "monthly_p": 50,
    "weekly_p": 150,
    "daily_p": 50,
}

# Which horizon block each model should read from in X
MODEL_FEATURE_BLOCK = {
    "monthly": "monthly",
    "weekly": "weekly",
    "daily": "daily",
    "monthly_p": "monthly",
    "weekly_p": "weekly",
    "daily_p": "daily",
}

# Individual threshold grids; all combinations are evaluated.
# Classification models use predict_proba thresholds, prediction models use predicted-return thresholds.
MODEL_THRESHOLD_GRIDS = {
    "monthly": [0.535],    # Manually set to 0.55
    "weekly": [0.535],     # Manually set to 0.52
    "daily": [0.535],
    "monthly_p": [-0.002],
    "weekly_p": [-0.002],
    "daily_p": [-0.002],
}

CLASSIFICATION_MODELS = ("monthly", "weekly", "daily")
PREDICTION_MODELS = ("monthly_p", "weekly_p", "daily_p")
VAL_MONTHS = 72   # last 24 months of training used for validation
TEST_MONTHS = 72  

def get_signal(bundle, X_raw_data, is_reg):
    lags = bundle['lags']
    X_slice = X_raw_data[:, :lags]  # Slices raw features based on model's specific lookback
    X_scaled = bundle['scaler'].transform(X_slice)
    if is_reg:
        return bundle['model'].predict(X_scaled)
    else:
        # Returns the probability of the positive class (column 1)
        return bundle['model'].predict_proba(X_scaled)[:, 1]

def cache_load(name):
    """Load cached object if exists, else return None."""
    path = CACHE_DIR / f"{name}.joblib"
    if path.exists():
        print(f"🔄 Loading cached {name} from {path}")
        return joblib.load(path)
    return None

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

def load_model(path):
    bundle = joblib.load(path)
    lags = bundle["lags"]
    print(f"{path}, {lags}")
    # Check sklearn version
    if "sklearn_version" in bundle:
        if bundle["sklearn_version"] != sklearn.__version__:
            raise RuntimeError(
                f"Model {path.name} was trained with sklearn {bundle['sklearn_version']}, "
                f"current sklearn={sklearn.__version__}"
            )
    else:
        print(f"⚠️ No sklearn version metadata in {path.name} — cannot verify compatibility")
    return bundle


def incremental_predict_with_partial_fit(
    loaded_models, X, y, d,
    train_core_mask, val_mask, test_mask,
    block_lags, offsets
):
    """
    Walk through the test period one month at a time.

    For each month t:
      1. Predict scores for all stocks in month t using the current model state.
      2. After prediction, call partial_fit on the data from month t so the
         model learns from that one new 'blink' of information before moving on.

    The scaler is NOT re-fitted — it was fitted on the training set and is
    kept frozen, which is the correct approach for online/incremental learning.

    Returns:
        model_scores : dict  {model_name -> np.ndarray of length n_test}
        The array order matches X[test_mask], i.e. the full test block sorted
        by date (same ordering as before).
    """
    dates = pd.to_datetime(d)
    if getattr(dates, "tz", None) is not None:
        dates = dates.tz_localize(None)

    test_dates = dates[test_mask]
    sorted_months = sorted(test_dates.unique())

    # Pre-allocate output arrays (same shape as the flat test block)
    n_test = test_mask.sum()
    model_scores = {name: np.full(n_test, np.nan) for name in loaded_models}

    # Build a flat index map: position in the test block for each row
    # test_indices[i] = position of the i-th test sample inside the full array
    test_indices_global = np.where(test_mask)[0]

    # Map global index → position within test block
    global_to_test_pos = {g: pos for pos, g in enumerate(test_indices_global)}

    # Classification classes needed for partial_fit
    CLASSES = np.array([0, 1])

    print(f"\n📅 Incremental test loop over {len(sorted_months)} months …")

    for month_idx, month in enumerate(sorted_months):
        # Boolean mask inside the FULL dataset for this month's test rows
        month_mask_global = test_mask & (dates == month)
        global_rows = np.where(month_mask_global)[0]

        if len(global_rows) == 0:
            continue

        # ── Step 1: PREDICT with current model state ──────────────────────
        for model_name, bundle in loaded_models.items():
            model      = bundle["model"]
            scaler     = bundle["scaler"]
            model_lag  = MODEL_LAGS[model_name]
            block_name = MODEL_FEATURE_BLOCK[model_name]
            start      = offsets[block_name]

            X_month = X[global_rows, start : start + model_lag]
            X_scaled = scaler.transform(X_month)

            if model_name in CLASSIFICATION_MODELS:
                preds = model.predict_proba(X_scaled)[:, 1]
            else:
                preds = model.predict(X_scaled)

            # Store in the flat test-block array at the correct positions
            for g_row, pred in zip(global_rows, preds):
                model_scores[model_name][global_to_test_pos[g_row]] = pred

        # ── Step 2: PARTIAL_FIT on this month's data ──────────────────────
        for model_name, bundle in loaded_models.items():
            model      = bundle["model"]
            scaler     = bundle["scaler"]
            model_lag  = MODEL_LAGS[model_name]
            block_name = MODEL_FEATURE_BLOCK[model_name]
            start      = offsets[block_name]

            X_month = X[global_rows, start : start + model_lag]
            X_scaled = scaler.transform(X_month)
            y_month  = y[global_rows]

            if model_name in CLASSIFICATION_MODELS:
                model.partial_fit(X_scaled, y_month, classes=CLASSES)
            else:
                # Regression models (MLPRegressor) also support partial_fit
                model.partial_fit(X_scaled, y_month)

        if (month_idx + 1) % 12 == 0:
            print(f"  … processed {month_idx + 1} / {len(sorted_months)} months")

    print("✅ Incremental loop complete.\n")
    return model_scores


def main():

    # Suppress the sklearn InconsistentVersionWarning globally during loading
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

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
        print(f"Loading {name} from {path.name} ...")
        loaded_models[name] = load_model(path)

    # Unpack into your original variable names
    monthly   = loaded_models["monthly"]
    weekly    = loaded_models["weekly"]
    daily     = loaded_models["daily"]
    monthly_p = loaded_models["monthly_p"]
    weekly_p  = loaded_models["weekly_p"]
    daily_p   = loaded_models["daily_p"]

    # Re-enable warnings later if needed
    warnings.filterwarnings("default", category=InconsistentVersionWarning)
    supervised = cache_load("supervised_arrays_class")
    X, y, d, R, stocks = supervised
    train_core_mask, val_mask, test_mask = split_masks(d, val_months=VAL_MONTHS, test_months=TEST_MONTHS)
    R = R[test_mask]

    block_lags = {
        block_name: max(
            MODEL_LAGS[model_name]
            for model_name, assigned_block in MODEL_FEATURE_BLOCK.items()
            if assigned_block == block_name
        )
        for block_name in ("monthly", "weekly", "daily")
    }

    offsets = {}
    cursor = 0
    for block_name in ("monthly", "weekly", "daily"):
        offsets[block_name] = cursor
        cursor += block_lags[block_name]

    # ── INCREMENTAL PREDICTION + PARTIAL_FIT ──────────────────────────────
    # Instead of scoring the entire test block at once, we walk month-by-month:
    #   predict first, then update with partial_fit on that month's observations.
    model_scores = incremental_predict_with_partial_fit(
        loaded_models, X, y, d,
        train_core_mask, val_mask, test_mask,
        block_lags, offsets
    )
    # ──────────────────────────────────────────────────────────────────────

    p_m, p_w, p_d = (
        model_scores["monthly"],
        model_scores["weekly"],
        model_scores["daily"],
    )
    X_mp, X_wp, X_dp = (
        model_scores["monthly_p"],
        model_scores["weekly_p"],
        model_scores["daily_p"],
    )

    results = []
    y_test = y[test_mask]
    ordered_models = list(MODEL_THRESHOLD_GRIDS.keys())
    threshold_value_lists = [MODEL_THRESHOLD_GRIDS[m] for m in ordered_models]
    for combo in itertools.product(*threshold_value_lists):
        threshold_map = dict(zip(ordered_models, combo))
        binary_preds = {
            model_name: (model_scores[model_name] >= threshold_map[model_name]).astype(int)
            for model_name in ordered_models
        }
        signal_mask = np.logical_and.reduce(
            [model_scores[model_name] >= threshold_map[model_name] for model_name in ordered_models]
        )

        n_signals = signal_mask.sum()
        if n_signals == 0:
            continue
        realized_returns = R[signal_mask]
        # strict consensus (AND rule)
        y_pred_combined = np.logical_and.reduce(
            [binary_preds[model_name] == 1 for model_name in ordered_models]
        ).astype(int)

        print(f"\n=== RESULTS with thresholds {threshold_map} ===")
        print(classification_report(y_test, y_pred_combined, digits=4))
        # Optional: show agreement rate
        agreement = np.mean(y_pred_combined == y_test)
        print(f"✅ Agreement with true labels: {agreement*100:.2f}%")
        row = {
            "coverage": n_signals / len(y_test),
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
        }
        for model_name in ordered_models:
            row[f"thr_{model_name}"] = threshold_map[model_name]
        results.append(row)

    df_results = pd.DataFrame(results).sort_values("mean_return", ascending=False).reset_index(drop=True)
    df_results.to_csv("df_results.csv", index=False)

    print("\n=== Threshold evaluation results ===")
    print(df_results)
    best_thresholds = {
        model_name: df_results.iloc[0][f"thr_{model_name}"] for model_name in ordered_models
    }
    print(f"\nUsing best threshold combination for ranking: {best_thresholds}")
    stocks = joblib.load("artifacts/stocks.joblib")

    # Validation slices
    dates_test  = pd.to_datetime(d[test_mask])
    stocks_test = stocks[test_mask]

    p_m_test, p_w_test, p_d_test = p_m, p_w, p_d
    X_mp_test, X_wp_test, X_dp_test = X_mp, X_wp, X_dp

    # Hyperparameters selected from sweep
    thr_monthly = best_thresholds["monthly"]
    thr_weekly = best_thresholds["weekly"]
    thr_daily = best_thresholds["daily"]
    thr_monthly_p = best_thresholds["monthly_p"]
    thr_weekly_p = best_thresholds["weekly_p"]
    thr_daily_p = best_thresholds["daily_p"]

    w_ret   = 1.0
    w_proba = 1.0

    monthly_rankings = {}
    var_ex = statistics.variance(X_mp_test)
    print(f"exact: {np.sqrt(var_ex)}")
    var_prob = statistics.variance(p_m_test)
    print(f"proba: {np.sqrt(var_prob)}")
    factor = np.sqrt(var_prob)/np.sqrt(var_ex) ## Note only calculated based on monthly variance difference
    print(f"factor: {factor}")

    for month in sorted(dates_test.unique()):
        idx = dates_test == month

        if idx.sum() == 0:
            continue

        # --- Binary mask (All 6 criteria must be met) ---
        mask = (
            (p_m_test[idx] >= thr_monthly) &
            (p_w_test[idx] >= thr_weekly) &
            (p_d_test[idx] >= thr_daily) &
            (X_mp_test[idx] >= thr_monthly_p) &
            (X_wp_test[idx] >= thr_weekly_p) &
            (X_dp_test[idx] >= thr_daily_p)
        )

        # If no stocks meet all 6 criteria for this month, skip to the next month
        if mask.sum() == 0:
            continue

        # --- Aggregate predictions (Only for the strict mask) ---
        avg_ret = (
            X_mp_test[idx][mask] +
            X_wp_test[idx][mask] +
            X_dp_test[idx][mask]
        ) / 3.0

        avg_proba = (
            p_m_test[idx][mask] +
            p_w_test[idx][mask] +
            p_d_test[idx][mask]
        ) / 3.0

        # --- Ranking score ---
        avg_prediction_threshold = np.mean([thr_monthly_p, thr_weekly_p, thr_daily_p])
        avg_classification_threshold = np.mean([thr_monthly, thr_weekly, thr_daily])
        
        score = (
            w_ret * (avg_ret - avg_prediction_threshold) +
            w_proba * ((avg_proba - avg_classification_threshold) / factor)
        )

        # Build dataframe for the ranking
        df_month = pd.DataFrame({
            "stock": stocks_test[idx][mask],
            "avg_pred_logret": avg_ret,
            "avg_proba": avg_proba,
            "score": score,
        }).sort_values("score", ascending=False)

        # Store the list of stocks that passed all 6 models
        monthly_rankings[month] = df_month["stock"].tolist()
    # Example output
    first_month = list(monthly_rankings.keys())[0]
    print(f"\nTop stocks for {first_month.date()}:")
    print(monthly_rankings[first_month][:10])
    print(len(monthly_rankings[first_month]))
    months = sorted(monthly_rankings.keys())


    # Optional: print all 72 lists
    #for m, tickers in zip(months, top_10_per_month):
        #print(f"\nTop stocks for {m.date()}: {tickers}")
    portfolio5 = []
    portfolio5.append(2000)
    portfolio10 = []
    portfolio10.append(2000)
    portfolio1 = []
    portfolio1.append(2000)
    i=0
    through_mask_n = []
    returns_5 = []
    returns_10 = []
    ret_1 = []
    for m in months:
        i+=1
        through_mask_n.append(len(monthly_rankings[m]))
        if len(monthly_rankings[m])>=10:
            picks = monthly_rankings[m][:10]
            picked_stocks = monthly_rankings[m][:5]

        elif len(monthly_rankings[m])>=5:
            picked_stocks = monthly_rankings[m][:5]
            picks = monthly_rankings[m]
        else:
            picked_stocks = monthly_rankings[m]
        pick = monthly_rankings[m][0]
        winningsm = 0
        winning = 0
        winningss = 0
        for stock in picked_stocks:
            stocks_test = stocks[test_mask]
            mask = (stocks_test == stock) & (dates_test == m)
            if np.any(mask):
                stock_return = R[mask][0]  # get the single matching return
                #print(f"{stock} on {dates_val}: {stock_return}")
            else:
                #print(f"No data for {stock} on {dates_val}")
                stock_p_l_5 = 0
            returns = math.exp(stock_return)
            stock_p_l_5 = (portfolio5[i-1]/len(picked_stocks))*(returns-1)
            winningsm += stock_p_l_5
        r = winningsm/portfolio5[i-1]
        returns_5.append(r)
        portfolio5.append(portfolio5[i-1] + winningsm) 
        for stock in picks:
            stocks_test = stocks[test_mask]
            mask = (stocks_test == stock) & (dates_test == m)
            if np.any(mask):
                stock_return = R[mask][0]  # get the single matching return
                #print(f"{stock} on {dates_val}: {stock_return}")
            else:
                #print(f"No data for {stock} on {dates_val}")
                stock_p_l_10 = 0
            returns = math.exp(stock_return)
            stock_p_l_10 = (portfolio10[i-1]/len(picks))*(returns-1)
            winningss += stock_p_l_10
        rs = winningss/portfolio10[i-1]
        returns_10.append(rs)
        portfolio10.append(portfolio10[i-1]+winningss)
        stock_test = stocks[test_mask]
        single_mask = (stock_test == pick) & (dates_test == m)
        if np.any(single_mask):
            stock_return_top = R[single_mask][0]  # get the single matching return
            #print(f"{pick} on {dates_val}: {stock_return_top}")
        else:
            #print(f"No data for {pick} on {dates_val}")
            stock_return_top = 0
        portfolio1.append(portfolio1[i-1] * math.exp(stock_return_top))
        ret_1.append(stock_return_top)
    E_R = 12*np.mean(returns_5)
    sigma_R = np.sqrt(12)*np.std(returns_5,ddof=1)
    max_drawdown = np.min(returns_5)
    r_f = 0.0191 ##ESTIMATE
    sharpe = (E_R-r_f)/sigma_R
    print(f"max_drawdown 5 stocks: {max_drawdown}")
    print(f"Sharpe ratio 5 stocks: {sharpe}")
    sh = (12*np.mean(returns_10)-0.0191)/(np.sqrt(12)*np.std(returns_10,ddof=1))
    max_d = np.min(returns_10)
    print(f"sharpe ratio 10 stocks: {sh}")
    print(f"max drawdown 10 stocks: {max_d}") 
    s = (12*np.mean(ret_1)-0.0191)/(np.sqrt(12)*np.std(ret_1,ddof=1))
    max_draw = np.min(ret_1)
    print(f"sharpe 1 stock: {s}")
    print(f"Max drawdown 1 stock: {max_draw}")
    plt.hist(through_mask_n, bins =100)
    plt.show()
    plt.plot(through_mask_n)
    plt.show()
    initial_month = pd.Timestamp("2019-09-30")
    months_with_initial = [initial_month] + [m.tz_localize(None) for m in months]

    portfolio_values = np.array(portfolio5)  # list → array
    portfolio_returns = portfolio_values[1:] / portfolio_values[:-1] - 1  # monthly simple returns
    portfolio_10_values = np.array(portfolio10)  # list → array
    portfolio_returns_10 = portfolio_10_values[1:] / portfolio_10_values[:-1] - 1
    portfolio_values_1 = np.array(portfolio1)  # list → array
    portfolio_returns_1 = portfolio_values_1[1:] / portfolio_values_1[:-1] - 1
    


    ticker = yf.Ticker("^GSPC")

    sp500 = ticker.history(
        start="2018-01-01",
        end="2025-01-01",
        auto_adjust=True
    )
    sp500.index = sp500.index.tz_localize(None)

    sp500 = sp500["Close"]
    sp500.index = pd.to_datetime(sp500.index).tz_localize(None)
    sp500 = sp500.sort_index()
    sp500 = sp500[~sp500.index.duplicated(keep="last")]

    months_clean = (
        pd.to_datetime(months_with_initial)
        .tz_localize(None)
        .to_period("M")
        .to_timestamp()
    )

    sp500_aligned = sp500.reindex(months_clean, method="ffill")
    sp500_returns = sp500_aligned.pct_change().fillna(0).values
    sharpe_S = (12*np.mean(sp500_returns)-(0.0191))/(np.sqrt(12)*np.std(sp500_returns, ddof=1))
    print(f"sharpe S&P: {sharpe_S}")
    print(f" Max draw S&P: {np.min(sp500_returns)}")

    # --- Compute cumulative returns ---
    cum_portfolio = np.cumprod(1 + portfolio_returns)
    # prepend initial value 1 (starting portfolio)
    cum_portfolio = np.insert(cum_portfolio, 0, 1.0)
    cum_portfolio_10 = np.cumprod(1 + portfolio_returns_10)
    # prepend initial value 1 (starting portfolio)
    cum_portfolio_10 = np.insert(cum_portfolio_10, 0, 1.0)
    cum_portfolio_1 = np.cumprod(1 + portfolio_returns_1)
    # prepend initial value 1 (starting portfolio)
    cum_portfolio_1 = np.insert(cum_portfolio_1, 0, 1.0)

    cum_sp500 = np.cumprod(1 + sp500_returns)
    plt.figure(figsize=(12,6))
    plt.plot(months_clean, cum_portfolio_1, label="Top 1 pick")
    plt.plot(months_clean, cum_portfolio, label="Top 5 picks")
    plt.plot(months_clean, cum_portfolio_10, label="Top 10 picks")
    plt.plot(months_clean, cum_sp500, label="S&P 500")
    plt.xticks(rotation=45)
    plt.title("Cumulative Returns: Strategy vs S&P 500")
    plt.xlabel("Month")
    plt.ylabel("Cumulative Return")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()