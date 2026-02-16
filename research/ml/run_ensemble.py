"""run_ensemble.py — Multi-strategy ensemble signal aggregation.

Loads multiple trained models, generates independent signals,
and combines them into a weighted ensemble decision. Allocates
capital based on recent performance and cross-strategy correlation.

Research Gap Fix #6: The original research focused on a single RSI
strategy with no diversification or ensemble framework.

Directive: Ensemble Strategy Framework.md
"""

import sys
import tomllib
from decimal import Decimal
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / ".tmp" / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_ensemble_config() -> dict:
    """Load ensemble configuration from config/ensemble.toml."""
    config_path = PROJECT_ROOT / "config" / "ensemble.toml"
    if not config_path.exists():
        print(f"ERROR: {config_path} not found.")
        sys.exit(1)
    with open(config_path, "rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------------
# Strategy loading
# ---------------------------------------------------------------------------


class Strategy:
    """Represents a single strategy in the ensemble.

    Attributes:
        name: Human-readable strategy label.
        model: Trained ML model (sklearn-compatible).
        weight: Capital allocation weight (0.0–1.0).
        config: Strategy-specific configuration.
        signals: Generated signal history.
    """

    def __init__(self, name: str, model_path: Path, weight: float, config: dict):
        """Initialise from model file.

        Args:
            name: Strategy name.
            model_path: Path to .joblib model file.
            weight: Initial allocation weight.
            config: Strategy-specific TOML config.
        """
        self.name = name
        self.weight = Decimal(str(weight))
        self.config = config
        self.signals: list[int] = []

        if model_path.exists():
            self.model = joblib.load(model_path)
            print(f"  ✓ Loaded {name}: {model_path.name} (weight={weight:.0%})")
        else:
            self.model = None
            print(f"  ⚠️  {name}: model not found at {model_path}")

    def predict(self, features: pd.DataFrame) -> int:
        """Generate a trading signal from features.

        Args:
            features: Feature row(s) for prediction.

        Returns:
            Signal: 1 (buy), -1 (sell), 0 (hold).
        """
        if self.model is None:
            return 0

        pred = self.model.predict(features)
        signal = int(pred[-1]) if len(pred) > 0 else 0

        # Convert binary classification to directional signal
        if signal == 1:
            return 1  # Buy
        elif signal == 0:
            return -1  # Sell
        return 0  # Hold

    @property
    def is_active(self) -> bool:
        """Check if the strategy has a loaded model."""
        return self.model is not None


def load_strategies() -> list[Strategy]:
    """Load all active strategies from the ensemble config.

    Returns:
        List of Strategy objects with loaded models.
    """
    config = load_ensemble_config()
    strategies = []

    for strat_cfg in config.get("strategies", []):
        if not strat_cfg.get("active", False):
            continue

        model_path = MODELS_DIR / strat_cfg.get("model", "")
        strategy_config_path = PROJECT_ROOT / "config" / strat_cfg.get("config", "")

        strat_config = {}
        if strategy_config_path.exists():
            with open(strategy_config_path, "rb") as f:
                strat_config = tomllib.load(f)

        strategies.append(
            Strategy(
                name=strat_cfg["name"],
                model_path=model_path,
                weight=strat_cfg.get("weight", 0.0),
                config=strat_config,
            )
        )

    return strategies


# ---------------------------------------------------------------------------
# Ensemble logic
# ---------------------------------------------------------------------------


def compute_correlation_matrix(signal_history: dict[str, list[int]]) -> pd.DataFrame:
    """Compute pairwise correlation between strategy signals.

    Args:
        signal_history: Dict of strategy_name → list of historical signals.

    Returns:
        Correlation matrix DataFrame.
    """
    df = pd.DataFrame(signal_history)
    if len(df) < 10:
        return pd.DataFrame()
    return df.corr()


def rebalance_weights(
    strategies: list[Strategy],
    signal_history: dict[str, list[int]],
    correlation_threshold: float = 0.70,
) -> dict[str, Decimal]:
    """Rebalance strategy weights based on performance and correlation.

    Strategies that are too correlated have their combined weight
    reduced to avoid concentration risk.

    Args:
        strategies: List of active strategies.
        signal_history: Historical signals per strategy.
        correlation_threshold: Max acceptable pairwise correlation.

    Returns:
        Dictionary of strategy_name → new weight.
    """
    weights = {s.name: s.weight for s in strategies if s.is_active}

    # Check for high correlation
    corr = compute_correlation_matrix(signal_history)
    if not corr.empty:
        for i, s1 in enumerate(corr.columns):
            for j, s2 in enumerate(corr.columns):
                if i >= j:
                    continue
                if abs(corr.iloc[i, j]) > correlation_threshold:
                    print(f"  ⚠️  High correlation ({corr.iloc[i, j]:.2f}) between {s1} and {s2}")
                    # Reduce the lower-weight strategy
                    if weights.get(s1, Decimal("0")) < weights.get(s2, Decimal("0")):
                        weights[s1] = weights.get(s1, Decimal("0")) * Decimal("0.5")
                    else:
                        weights[s2] = weights.get(s2, Decimal("0")) * Decimal("0.5")

    # Normalise weights to sum to 1
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}

    return weights


def ensemble_signal(
    strategies: list[Strategy],
    features: pd.DataFrame,
    weights: dict[str, Decimal] | None = None,
) -> tuple[int, dict[str, int]]:
    """Generate a weighted ensemble signal.

    Args:
        strategies: List of active strategies.
        features: Feature DataFrame for prediction.
        weights: Optional weight overrides.

    Returns:
        Tuple of (ensemble_signal, individual_signals_dict).
    """
    individual_signals: dict[str, int] = {}
    weighted_sum = Decimal("0")

    for strategy in strategies:
        if not strategy.is_active:
            continue

        signal = strategy.predict(features)
        individual_signals[strategy.name] = signal

        w = weights.get(strategy.name, strategy.weight) if weights else strategy.weight
        weighted_sum += Decimal(str(signal)) * w

    # Threshold: need >0.3 weighted consensus to trade
    if weighted_sum > Decimal("0.3"):
        ensemble = 1  # Buy
    elif weighted_sum < Decimal("-0.3"):
        ensemble = -1  # Sell
    else:
        ensemble = 0  # Hold

    return ensemble, individual_signals


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the ensemble framework in analysis mode."""
    print("🎯 Ensemble Strategy Framework\n")

    config = load_ensemble_config()
    ensemble_cfg = config.get("ensemble", {})

    strategies = load_strategies()
    active = [s for s in strategies if s.is_active]

    min_strats = ensemble_cfg.get("min_strategies", 2)
    if len(active) < min_strats:
        print(f"\n⚠️  Only {len(active)} active strategies (minimum {min_strats} required).")
        print("   Train more strategy models before using the ensemble.\n")
        print("   Registered strategies:")
        for strat_cfg in config.get("strategies", []):
            status = "✓ active" if strat_cfg.get("active") else "○ inactive"
            print(f"     {strat_cfg['name']:30s} {status}")
        sys.exit(0)

    print(f"\n  Active: {len(active)}/{len(strategies)} strategies")
    print(f"  Rebalance: {ensemble_cfg.get('rebalance_frequency', 'weekly')}")
    print(f"  Correlation cap: {ensemble_cfg.get('correlation_threshold', 0.70)}\n")

    # Check for feature data
    features_path = DATA_DIR / "features" / "X.parquet"
    if not features_path.exists():
        print("  No feature data found. Run build_ml_features.py first.")
        print("  (Running in config-check mode only.)\n")
        return

    # Demo: run ensemble on latest feature row
    X = pd.read_parquet(features_path)
    latest = X.tail(1)

    ensemble, individual = ensemble_signal(active, latest)
    signal_label = {1: "📈 BUY", -1: "📉 SELL", 0: "⏸️ HOLD"}

    print(f"  Ensemble Signal: {signal_label.get(ensemble, 'UNKNOWN')}")
    print("  Individual:")
    for name, sig in individual.items():
        print(f"    {name:30s} → {signal_label.get(sig, 'UNKNOWN')}")

    print("\n✅ Ensemble analysis complete.\n")


if __name__ == "__main__":
    main()
