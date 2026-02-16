"""run_ml_strategy.py — End-to-end ML Strategy Discovery Pipeline.

Pipeline steps:
  1. Load EUR/USD data from data/ across multiple timeframes
  2. Build feature matrix (technical indicators + MTF confluence)
  3. Engineer 3-class target: LONG (+1), SHORT (-1), FLAT (0)
  4. Train models via walk-forward expanding-window CV
  5. Backtest ML predictions via VectorBT (long + short)
  6. Report metrics and save model if profitable

Directive: Machine Learning Strategy Discovery.md
"""

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import vectorbt as vbt
except ImportError:
    vbt = None
    print("  ⚠ vectorbt not installed — VBT backtest will be skipped.")

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

try:
    import xgboost as xgb
except ImportError:
    xgb = None
    print("  ⚠ xgboost not installed — XGBoost model will be skipped.")


# ─────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────


def load_ohlcv(pair: str, gran: str) -> pd.DataFrame | None:
    """Load Parquet OHLCV data, return None if missing."""
    path = DATA_DIR / f"{pair}_{gran}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df


# Import shared feature engineering logic
from titan.strategies.ml.features import (
    atr,
    build_features,
)


def build_target(
    close: pd.Series, atr_series: pd.Series, tp_mult: float = 1.5, sl_mult: float = 1.0
) -> pd.Series:
    """Build a 3-class directional target: 0 SHORT, 1 FLAT, 2 LONG.

    Uses ATR-based forward return thresholds to filter out noise.
    A move must exceed tp_mult × ATR to count as a valid signal.

    Args:
        close: Close price series.
        atr_series: ATR series for dynamic thresholds.
        tp_mult: ATR multiplier for the signal threshold.
        sl_mult: ATR multiplier for the stop threshold (not used in labelling).

    Returns:
        Series of target labels: 0, 1, or 2.
    """
    fwd_return = close.shift(-1) / close - 1
    threshold = atr_series / close * tp_mult

    target = pd.Series(1, index=close.index, name="target")  # Default 1 (FLAT)
    target[fwd_return > threshold] = 2  # LONG
    target[fwd_return < -threshold] = 0  # SHORT

    return target


# ─────────────────────────────────────────────────────────────────────
# Walk-Forward Training
# ─────────────────────────────────────────────────────────────────────


def walk_forward_splits(n: int, n_splits: int = 5, min_train: float = 0.4):
    """Generate expanding-window walk-forward splits."""
    min_tr = int(n * min_train)
    test_size = (n - min_tr) // n_splits
    splits = []
    for i in range(n_splits):
        tr_end = min_tr + i * test_size
        te_end = min(tr_end + test_size, n)
        if te_end > tr_end:
            splits.append((np.arange(tr_end), np.arange(tr_end, te_end)))
    return splits


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, close: pd.Series):
    """Train ML models with walk-forward CV and return the best one."""
    from sklearn.metrics import accuracy_score

    models = {
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
        ),
    }

    # Try to add XGBoost
    if xgb is not None:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            use_label_encoder=False,
        )
    else:
        print("  ⚠ XGBoost not installed — skipping.")

    splits = walk_forward_splits(len(X), n_splits=5)
    best_model = None
    best_name = ""
    best_sharpe = -np.inf
    results = {}

    print(f"\n  {'─' * 60}")
    print(f"  [TRAIN] Training {len(models)} models × 5 walk-forward folds")
    print(f"  {'─' * 60}\n")

    for name, model in models.items():
        fold_sharpes = []
        fold_accs = []
        all_preds = []
        all_true = []

        for fold_i, (train_idx, test_idx) in enumerate(splits):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)

            acc = accuracy_score(y_te, y_pred)
            fold_accs.append(acc)

            # Signal Sharpe: predicted direction × actual return
            # Map back: 0 -> -1, 1 -> 0, 2 -> 1
            pred_mapped = np.where(y_pred == 2, 1, np.where(y_pred == 0, -1, 0))
            fwd_ret = close.pct_change().shift(-1).iloc[test_idx]
            strat_ret = pd.Series(pred_mapped, index=fwd_ret.index) * fwd_ret
            if strat_ret.std() > 0:
                sharpe = float(strat_ret.mean() / strat_ret.std() * np.sqrt(252 * 6))
            else:
                sharpe = 0.0
            fold_sharpes.append(sharpe)

            all_preds.extend(y_pred)
            all_true.extend(y_te)

        avg_sharpe = np.mean(fold_sharpes)
        avg_acc = np.mean(fold_accs)

        # Class distribution in predictions
        pred_arr = np.array(all_preds)
        n_long = (pred_arr == 2).sum()  # 2 is LONG
        n_short = (pred_arr == 0).sum()  # 0 is SHORT
        n_flat = (pred_arr == 1).sum()  # 1 is FLAT
        total = len(pred_arr)

        print(
            f"  {name:25s}  Sharpe={avg_sharpe:>7.3f}  "
            f"Acc={avg_acc:.3f}  "
            f"L={n_long}/{total} S={n_short}/{total} F={n_flat}/{total}"
        )

        results[name] = {
            "sharpe": avg_sharpe,
            "accuracy": avg_acc,
            "fold_sharpes": fold_sharpes,
            "n_long": int(n_long),
            "n_short": int(n_short),
            "n_flat": int(n_flat),
        }

        if avg_sharpe > best_sharpe:
            best_sharpe = avg_sharpe
            best_name = name
            best_model = model

    # Retrain best model on full dataset
    print(f"\n  🏆 Best model: {best_name} (Sharpe={best_sharpe:.3f})")
    best_model.fit(X, y)

    # Feature importance
    importances = best_model.feature_importances_
    imp_df = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    print("\n  📊 Top 15 Features:")
    for _, row in imp_df.head(15).iterrows():
        bar = "█" * int(row["importance"] * 200)
        print(f"     {row['feature']:20s} {row['importance']:.4f} {bar}")

    return best_model, best_name, results, imp_df


# ─────────────────────────────────────────────────────────────────────
# VBT Backtest
# ─────────────────────────────────────────────────────────────────────


def optimize_exits(
    close: pd.Series,
    preds: np.ndarray,
    stop_values: list[float] = [0.005, 0.01, 0.015, 0.02],
    freq: str = "4h",
) -> dict:
    """Optimize exit strategy (Fixed vs Trailing Stop) using OOS signals."""
    if vbt is None:
        return {}

    # 2 is LONG, 0 is SHORT
    long_entries = pd.Series(preds == 2, index=close.index)
    short_entries = pd.Series(preds == 0, index=close.index)

    # We will close on opposite signal as a base, but stops will override
    long_exits = pd.Series(preds == 0, index=close.index)
    short_exits = pd.Series(preds == 2, index=close.index)

    results = []
    print(f"\n  {'─' * 60}")
    print(f"  🏁 Exit Optimization (OOS) - Freq: {freq}")
    print(f"  {'─' * 60}")
    print(f"  {'Type':<10} {'Stop':<8} {'Sharpe':<8} {'Return':<8} {'Trades':<6}")

    for sl in stop_values:
        # 1. Fixed Stop Loss / Take Profit (Set TP = 2 * SL for 1:2 R:R)
        pf_fixed = vbt.Portfolio.from_signals(
            close,
            entries=long_entries,
            exits=long_exits,
            short_entries=short_entries,
            short_exits=short_exits,
            sl_stop=sl,
            tp_stop=sl * 2.0,  # 1:2 Risk:Reward
            init_cash=10_000,
            fees=0.0002,
            freq=freq,
        )
        stats_fixed = {
            "type": "Fixed",
            "stop": sl,
            "sharpe": pf_fixed.sharpe_ratio(),
            "return": pf_fixed.total_return() * 100,
            "trades": int(pf_fixed.trades.count()),
        }
        results.append(stats_fixed)
        print(
            f"  {stats_fixed['type']:<10} {stats_fixed['stop']:<8.3%} "
            f"{stats_fixed['sharpe']:<8.3f} {stats_fixed['return']:<8.2f}% "
            f"{stats_fixed['trades']:<6}"
        )

        # 2. Trailing Stop Loss
        pf_trail = vbt.Portfolio.from_signals(
            close,
            entries=long_entries,
            exits=long_exits,
            short_entries=short_entries,
            short_exits=short_exits,
            sl_trail=sl,  # Trailing stop
            init_cash=10_000,
            fees=0.0002,
            freq=freq,
        )
        stats_trail = {
            "type": "Trailing",
            "stop": sl,
            "sharpe": pf_trail.sharpe_ratio(),
            "return": pf_trail.total_return() * 100,
            "trades": int(pf_trail.trades.count()),
        }
        results.append(stats_trail)
        print(
            f"  {stats_trail['type']:<10} {stats_trail['stop']:<8.3%} "
            f"{stats_trail['sharpe']:<8.3f} {stats_trail['return']:<8.2f}% "
            f"{stats_trail['trades']:<6}"
        )

    # Find best
    if not results:
        return {}

    best = max(results, key=lambda x: x["sharpe"])
    print(f"\n  🏆 Best Exit: {best['type']} Stop {best['stop']:.1%} (Sharpe {best['sharpe']:.3f})")

    return best


def backtest_ml_predictions(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    close: pd.Series,
    split_pct: float = 0.70,
    freq: str = "4h",
):
    """Run IS/OOS backtest of the ML model's predictions via VBT."""
    if vbt is None:
        print("  ⚠ VBT not available — skipping backtest.")
        return {}

    split = int(len(X) * split_pct)

    # Train on IS, predict on OOS
    X_is, X_oos = X.iloc[:split], X.iloc[split:]
    _close_is, close_oos = close.iloc[:split], close.iloc[split:]
    _y_is = y.iloc[:split]  # noqa: F841

    model.fit(X_is, y.iloc[:split])
    preds_oos = model.predict(X_oos)

    # 1. Base Backtest (Signal Only)
    n_l = (preds_oos == 2).sum()
    n_s = (preds_oos == 0).sum()
    n_f = (preds_oos == 1).sum()
    print(f"    OOS predictions: LONG={n_l}  SHORT={n_s}  FLAT={n_f}")

    long_entries = pd.Series(preds_oos == 2, index=close_oos.index)
    long_exits = pd.Series(preds_oos != 2, index=close_oos.index)
    short_entries = pd.Series(preds_oos == 0, index=close_oos.index)
    short_exits = pd.Series(preds_oos != 0, index=close_oos.index)

    long_pf = vbt.Portfolio.from_signals(
        close_oos,
        entries=long_entries,
        exits=long_exits,
        init_cash=10_000,
        fees=0.0002,
        freq=freq,
    )
    short_pf = vbt.Portfolio.from_signals(
        close_oos,
        entries=pd.Series(False, index=close_oos.index),
        exits=pd.Series(False, index=close_oos.index),
        short_entries=short_entries,
        short_exits=short_exits,
        init_cash=10_000,
        fees=0.0002,
        freq=freq,
    )

    print(f"\n  {'─' * 60}")
    print("  📈 VBT OOS Backtest (Base Signals)")
    print(f"  {'─' * 60}")

    for label, pf in [("LONG", long_pf), ("SHORT", short_pf)]:
        ret = pf.total_return() * 100
        sharpe = pf.sharpe_ratio()
        dd = pf.max_drawdown() * 100
        trades = pf.trades.count()
        wr = pf.trades.win_rate() * 100 if trades > 0 else 0
        print(
            f"    {label:6s}  Return={ret:>7.2f}%  Sharpe={sharpe:>7.3f}  "
            f"MaxDD={dd:>6.2f}%  Trades={trades}  WR={wr:.1f}%"
        )

    # 2. Exit Optimization
    best_exit = optimize_exits(close_oos, preds_oos, freq=freq)
    return best_exit


# ─────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────


def run_pipeline(base_tf: str = "H4"):
    """Run the full ML pipeline for a specific base timeframe."""
    print(f"\n{'=' * 60}")
    print(f"  [INFO] ML Strategy Discovery — EUR_USD {base_tf}")
    print(f"{'=' * 60}")

    # ── Step 1: Load data ──
    print("\n  Step 1: Loading data...")
    _data_dir = Path("data")  # noqa: F841

    # Load base timeframe
    df = load_ohlcv("EUR_USD", base_tf)
    if df is None:
        print(f"  ❌ Error: Base data {base_tf} not found.")
        return {}
    df = df.sort_index()

    # Load context timeframes (Higher TFs)
    # If H1, context is H4, D, W
    # If H4, context is D, W
    context_data = {}
    possible_contexts = ["H4", "D", "W"]

    # Only load contexts that are higher than base_tf
    # Simple hierarchy: H1 < H4 < D < W
    hierarchy = {"H1": 0, "H4": 1, "D": 2, "W": 3}
    base_rank = hierarchy.get(base_tf, 0)

    for tf in possible_contexts:
        if hierarchy[tf] > base_rank:
            ctx_df = load_ohlcv("EUR_USD", tf)
            if ctx_df is not None:
                context_data[tf] = ctx_df.sort_index()
                print(f"    Loaded context: {tf} ({len(context_data[tf])} bars)")

    print(f"    Base: {base_tf} ({len(df)} bars) {df.index.min().date()} → {df.index.max().date()}")

    # ── Step 2: Build features ──
    print("\n  Step 2: Building feature matrix...")
    feats = build_features(df, context_data)
    print(f"    {len(feats.columns)} features built")

    # ── Step 3: Build target ──
    print("\n  Step 3: Engineering target (3-class: LONG/SHORT/FLAT)...")
    close = df["close"]
    atr_series = atr(df, 14)
    # Lower threshold to 0.5 ATR to catch more moves
    target = build_target(close, atr_series, tp_mult=0.5, sl_mult=0.5)

    # Align features and target
    valid = feats.notna().all(axis=1) & target.notna() & atr_series.notna()
    feats = feats[valid]
    target = target[valid]
    close_aligned = close[valid]

    # Target distribution
    n_long = (target == 2).sum()
    n_short = (target == 0).sum()
    n_flat = (target == 1).sum()
    print(f"    Final dataset: {len(feats)} rows × {feats.shape[1]} features")
    print(
        f"    Target distribution:  LONG={n_long} ({n_long / len(target) * 100:.1f}%)  "
        f"SHORT={n_short} ({n_short / len(target) * 100:.1f}%)  "
        f"FLAT={n_flat} ({n_flat / len(target) * 100:.1f}%)"
    )

    if len(feats) < 100:
        print("  ⚠ Not enough data to train. Skipping.")
        return

    # ── Step 4: Train models ──
    print("\n  Step 4: Training ML models with walk-forward CV...")

    # Delegate to train_and_evaluate which handles the loop and best model selection
    best_model, best_model_name, results, imp_df = train_and_evaluate(feats, target, close_aligned)
    best_model_score = results[best_model_name]["sharpe"]

    print(f"\n  [BEST] Best model selected: {best_model_name} (Sharpe={best_model_score:.3f})")

    # ── Step 5: VBT Backtest ──
    print("\n  Step 5: Running VBT backtest on OOS period...")

    # Run backtest/exit optimization
    # Pass freq to ensure VBT calculates Sharpe correctly
    freq_map = {"H1": "1h", "H4": "4h"}
    vbt_freq = freq_map.get(base_tf, "1h")

    best_exit = backtest_ml_predictions(best_model, feats, target, close_aligned, freq=vbt_freq)

    # ── Step 6: Save model ──
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_fname = f"ml_strategy_{base_tf}_{best_model_name.lower()}_{version}.joblib"
    model_path = Path("models") / model_fname
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(best_model, model_path)
    print(f"\n  [SAVE] Model saved → {model_path.name}")

    # Save report
    report = {
        "timestamp": version,
        "base_tf": base_tf,
        "model": best_model_name,
        "sharpe_cv": best_model_score,
        "best_exit": best_exit,
        "features": list(feats.columns),
        "target_dist": {"LONG": int(n_long), "SHORT": int(n_short), "FLAT": int(n_flat)},
    }
    report_path = Path(".tmp/reports") / f"ml_strategy_{base_tf}_{version}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  [SAVE] Report saved → {report_path.name}")

    print(f"\n{'=' * 60}")
    print("  [DONE] ML Strategy Discovery Complete")
    print(f"{'=' * 60}\n")


def main():
    # H1 tends to overfit (OOS Sharpe < 0), but kept for comparison.
    target_timeframes = ["H4", "H1"]
    results = []

    for tf in target_timeframes:
        try:
            res = run_pipeline(tf)
            if res:
                results.append(res)
        except Exception as e:
            print(f"  [ERROR] Error processing {tf}: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print("  Multi-Timeframe Scan Complete")
    print(f"{'=' * 60}")

    print(f"  {'Timeframe':<10} {'Model':<18} {'Exit Type':<12} {'Stop':<8} {'Sharpe (OOS)':<12}")
    print(f"  {'-' * 60}")

    for res in results:
        exit_strat = res.get("best_exit", {})
        # Handle cases where best_exit might be empty or None
        if not exit_strat:
            continue

        print(
            f"  {res['base_tf']:<10} {res['model']:<18} "
            f"{exit_strat.get('type', 'N/A'):<12} "
            f"{exit_strat.get('stop', 0):<8.1%} "
            f"{exit_strat.get('sharpe', 0.0):<12.3f}"
        )


if __name__ == "__main__":
    main()
