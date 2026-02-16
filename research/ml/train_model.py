"""train_ml_model.py — Train an ML model with walk-forward validation.

Uses vbt.Splitter for expanding/rolling window cross-validation,
trains XGBClassifier or RandomForest, evaluates with confusion matrix
and Sharpe Ratio, and serialises to models/ if performance threshold met.

Directive: Machine Learning Strategy Discovery.md
"""

import json
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FEATURES_DIR = PROJECT_ROOT / ".tmp" / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------
def _get_model_registry() -> dict:
    """Lazy import of model classes to handle optional dependencies.

    Returns:
        Dictionary mapping model type names to their classes.
    """
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

    registry = {
        "GradientBoosting": GradientBoostingClassifier,
        "RandomForest": RandomForestClassifier,
    }

    try:
        from xgboost import XGBClassifier

        registry["XGBClassifier"] = XGBClassifier
    except ImportError:
        print("  ⚠️ xgboost not installed. XGBClassifier unavailable.")

    return registry


def load_training_config() -> dict:
    """Load training configuration from config/training.toml."""
    config_path = PROJECT_ROOT / "config" / "training.toml"
    if not config_path.exists():
        print(f"ERROR: {config_path} not found.")
        sys.exit(1)
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def walk_forward_splits(
    n_samples: int, n_splits: int = 5, min_train_pct: float = 0.3
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate expanding-window walk-forward train/test splits.

    Mimics real-time learning: the training window grows, the test window
    stays fixed. This is equivalent to vbt.Splitter's expanding window mode.

    Args:
        n_samples: Total number of samples.
        n_splits: Number of walk-forward folds.
        min_train_pct: Minimum training set size as fraction of total.

    Returns:
        List of (train_indices, test_indices) tuples.
    """
    min_train = int(n_samples * min_train_pct)
    test_size = (n_samples - min_train) // n_splits
    splits = []
    for i in range(n_splits):
        train_end = min_train + i * test_size
        test_end = min(train_end + test_size, n_samples)
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, test_end)
        if len(test_idx) > 0:
            splits.append((train_idx, test_idx))
    return splits


def compute_signal_sharpe(y_pred: np.ndarray, returns: pd.Series) -> float:
    """Compute the Sharpe Ratio of predicted signals.

    This evaluates the trading profitability of the model's predictions,
    not just classification accuracy.

    Args:
        y_pred: Binary predictions (1=long, 0=flat).
        returns: Actual forward returns aligned with predictions.

    Returns:
        Annualised Sharpe Ratio.
    """
    strategy_returns = pd.Series(y_pred, index=returns.index) * returns
    if strategy_returns.std() == 0:
        return 0.0
    return float(strategy_returns.mean() / strategy_returns.std() * np.sqrt(252))


def main() -> None:
    """Train ML model with walk-forward validation and Sharpe evaluation."""
    config = load_training_config()
    model_cfg = config.get("model", {})
    split_cfg = config.get("split", {})
    hp_cfg = config.get("hyperparameters", {})

    model_type = model_cfg.get("type", "XGBClassifier")
    random_state = model_cfg.get("random_state", 42)  # MANDATORY

    registry = _get_model_registry()
    if model_type not in registry:
        print(f"ERROR: Unknown model type '{model_type}'. Options: {list(registry.keys())}")
        sys.exit(1)

    # Load feature data
    feature_files = sorted(FEATURES_DIR.glob("*_features.parquet"))
    target_files = sorted(FEATURES_DIR.glob("*_target.parquet"))

    if not feature_files or not target_files:
        print("ERROR: No feature/target files. Run build_ml_features.py first.")
        sys.exit(1)

    X = pd.read_parquet(feature_files[0])
    y = pd.read_parquet(target_files[0]).squeeze()

    print(f"🧠 Training {model_type} (random_state={random_state})")
    print(f"   Dataset: {X.shape[0]} rows × {X.shape[1]} features\n")

    # Walk-forward splits
    n_splits = split_cfg.get("n_splits", 5)
    splits = walk_forward_splits(len(X), n_splits=n_splits)

    hp_cfg["random_state"] = random_state
    model_cls = registry[model_type]

    # Filter out incompatible hyperparameters for the chosen model
    import inspect

    valid_params = set(inspect.signature(model_cls.__init__).parameters.keys())
    filtered_hp = {k: v for k, v in hp_cfg.items() if k in valid_params}

    model = model_cls(**filtered_hp)

    print(f"   Walk-forward CV with {len(splits)} expanding folds...")
    fold_sharpes = []
    all_y_test = []
    all_y_pred = []

    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Compute fold metrics
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        acc = report["accuracy"]

        # Signal Sharpe (using return_lag_1 as proxy for returns)
        if "return_lag_1" in X.columns:
            returns = X.iloc[test_idx]["return_lag_1"]
            fold_sharpe = compute_signal_sharpe(y_pred, returns)
        else:
            fold_sharpe = 0.0

        fold_sharpes.append(fold_sharpe)
        all_y_test.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        print(f"   Fold {fold + 1}: accuracy={acc:.4f}  Sharpe={fold_sharpe:.4f}")

    avg_sharpe = np.mean(fold_sharpes)
    print(f"\n   📊 Average Signal Sharpe: {avg_sharpe:.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_y_test, all_y_pred)
    print(f"   Confusion Matrix:\n{cm}\n")

    # Final train on all data
    model.fit(X, y)

    # Feature importance
    importances = model.feature_importances_
    importance_df = pd.DataFrame(
        {"feature": X.columns.tolist(), "importance": importances}
    ).sort_values("importance", ascending=False)

    print("   Top 10 Features:")
    for _, row in importance_df.head(10).iterrows():
        bar = "█" * int(row["importance"] * 100)
        flag = " ⚠️ DOMINANT" if row["importance"] > 0.40 else ""
        print(f"     {row['feature']:25s} {row['importance']:.4f} {bar}{flag}")

    # Serialise if Sharpe > 1.5 threshold
    sharpe_threshold = 1.5
    if avg_sharpe >= sharpe_threshold:
        version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_path = MODELS_DIR / "production_model_v1.joblib"
        joblib.dump(model, model_path)
        print(f"\n   💾 Model saved to {model_path} (Sharpe {avg_sharpe:.4f} ≥ {sharpe_threshold})")
    else:
        print(
            f"\n   ⚠️  Sharpe {avg_sharpe:.4f} < {sharpe_threshold}. "
            "Model NOT saved. Iterate on features."
        )

    # Save training report
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"training_{version}.json"
    report_data = {
        "model_type": model_type,
        "random_state": random_state,
        "hyperparameters": hp_cfg,
        "n_splits": n_splits,
        "avg_signal_sharpe": avg_sharpe,
        "fold_sharpes": fold_sharpes,
        "confusion_matrix": cm.tolist(),
        "feature_importance": importance_df.to_dict(orient="records"),
    }
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2, default=str)
    print(f"   📄 Report saved to {report_path}")

    print("\n✅ Model training complete.\n")


if __name__ == "__main__":
    main()
