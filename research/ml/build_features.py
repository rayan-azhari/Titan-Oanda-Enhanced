"""build_ml_features.py — Build ML feature matrix and target vector.

Transforms raw OHLCV Parquet data into a Feature Matrix (X) and
Target Vector (y) suitable for ML model training. All features
use only past data to prevent look-ahead bias.

Directive: Machine Learning Strategy Discovery.md
"""

import sys
import tomllib
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DATA_DIR = PROJECT_ROOT / ".tmp" / "data" / "raw"
FEATURES_DIR = PROJECT_ROOT / ".tmp" / "data" / "features"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports"

FEATURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Technical Indicator Functions (all look-back only — no future data)
# ---------------------------------------------------------------------------
def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average.

    Args:
        series: Input price series.
        period: Look-back window size.

    Returns:
        SMA series.
    """
    return series.rolling(window=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average.

    Args:
        series: Input price series.
        period: Look-back span.

    Returns:
        EMA series.
    """
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index.

    Args:
        series: Input price series.
        period: RSI look-back period.

    Returns:
        RSI series (0–100).
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range.

    Args:
        df: DataFrame containing high, low, close columns.
        period: ATR look-back period.

    Returns:
        ATR series.
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD indicator: line, signal, and histogram.

    Args:
        series: Input price series.
        fast: Fast EMA period.
        slow: Slow EMA period.
        signal: Signal line EMA period.

    Returns:
        DataFrame with macd, macd_signal, macd_hist columns.
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": histogram})


def bollinger_bandwidth(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    """Bollinger Bandwidth (normalised).

    Args:
        series: Input price series.
        period: Look-back window.
        std_dev: Standard deviation multiplier.

    Returns:
        Normalised bandwidth series.
    """
    mid = sma(series, period)
    std = series.rolling(window=period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return (upper - lower) / mid


def time_of_day_features(timestamps: pd.Series) -> pd.DataFrame:
    """Extract time-of-day features (hour, session flags).

    Args:
        timestamps: Timestamp series.

    Returns:
        DataFrame with hour and trading session indicator columns.
    """
    hour = timestamps.dt.hour
    return pd.DataFrame(
        {
            "hour": hour,
            "is_london": ((hour >= 8) & (hour < 16)).astype(int),
            "is_new_york": ((hour >= 13) & (hour < 21)).astype(int),
            "is_tokyo": ((hour >= 0) & (hour < 9)).astype(int),
        }
    )


# ---------------------------------------------------------------------------
# Feature Matrix Builder
# ---------------------------------------------------------------------------
def build_feature_matrix(
    df: pd.DataFrame, config: dict, mtf_features: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the full feature matrix (X) and target vector (y).

    IMPORTANT: The target is shifted backwards by 1 period to prevent
    look-ahead bias. y[t] = sign(close[t+1] - close[t]).

    Args:
        df: Raw OHLCV DataFrame.
        config: Feature configuration dictionary from features.toml.

    Returns:
        Tuple of (feature_matrix, target_vector).
    """
    close = df["close"].astype(float)
    features = pd.DataFrame(index=df.index)

    # --- Lagged returns (always included) ---
    for lag in config.get("lags", [1, 2, 5, 10, 20]):
        features[f"return_lag_{lag}"] = close.pct_change(lag)

    # --- Trend indicators ---
    trend_cfg = config.get("trend", {})
    for period in trend_cfg.get("sma_periods", [20, 50]):
        features[f"sma_{period}"] = sma(close, period)
    for period in trend_cfg.get("ema_periods", [12, 26]):
        features[f"ema_{period}"] = ema(close, period)
    if trend_cfg.get("macd", True):
        macd_df = macd(close)
        features = pd.concat([features, macd_df], axis=1)

    # --- Momentum ---
    momentum_cfg = config.get("momentum", {})
    if momentum_cfg.get("rsi", True):
        features["rsi_14"] = rsi(close, 14)

    # --- Volatility ---
    vol_cfg = config.get("volatility", {})
    if vol_cfg.get("atr", True):
        features["atr_14"] = atr(df, 14)
    if vol_cfg.get("bollinger_bw", True):
        features["boll_bw_20"] = bollinger_bandwidth(close, 20)

    # --- Time-of-day features ---
    if "timestamp" in df.columns:
        tod = time_of_day_features(df["timestamp"])
        features = pd.concat([features, tod.set_index(features.index)], axis=1)

    # --- Multi-timeframe confluence features (if available) ---
    # These are pre-computed by mtf_confluence.py and aligned to H1 timeline.
    mtf_cols = [
        "h1_bias",
        "h4_bias",
        "d_bias",
        "confluence_score",
        "all_bullish",
        "all_bearish",
        "signal",
    ]
    if mtf_features is not None and not mtf_features.empty:
        for col in mtf_cols:
            if col in mtf_features.columns:
                aligned = mtf_features[col].reindex(features.index, method="ffill")
                features[f"mtf_{col}"] = aligned
        print(
            f"    Added {sum(1 for c in mtf_cols if c in mtf_features.columns)} "
            "MTF confluence features."
        )

    # --- Target vector (shifted backward by 1 to prevent look-ahead) ---
    # y[t] = 1 if close[t+1] > close[t], else 0
    target = (close.shift(-1) > close).astype(int)
    target.name = "target"

    # Drop warm-up NaN rows
    valid = features.notna().all(axis=1) & target.notna()
    features = features[valid]
    target = target[valid]

    n_dropped = len(df) - len(features)
    print(f"    Dropped {n_dropped} rows (warm-up NaN + tail).")

    return features, target


def load_features_config() -> dict:
    """Load features.toml configuration file."""
    config_path = PROJECT_ROOT / "config" / "features.toml"
    if not config_path.exists():
        print(f"WARNING: {config_path} not found. Using defaults.")
        return {}
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def main() -> None:
    """Build feature matrices for all raw data files."""
    config = load_features_config()

    raw_files = list(RAW_DATA_DIR.glob("*.parquet"))
    if not raw_files:
        print("ERROR: No raw data in .tmp/data/raw/. Run download_oanda_data.py first.")
        sys.exit(1)

    print(f"🔧 Building ML features for {len(raw_files)} dataset(s)\n")

    for raw_path in raw_files:
        print(f"  Processing {raw_path.name}...")
        df = pd.read_parquet(raw_path)

        # Load MTF confluence features if available
        pair_gran = raw_path.stem  # e.g. "EUR_USD_H4"
        pair = "_".join(pair_gran.split("_")[:2])  # e.g. "EUR_USD"
        mtf_path = FEATURES_DIR / f"{pair}_mtf_confluence.parquet"
        mtf_features = None
        if mtf_path.exists():
            mtf_features = pd.read_parquet(mtf_path)
            print(f"    Loading MTF confluence from {mtf_path.name}")

        features, target = build_feature_matrix(df, config, mtf_features=mtf_features)

        # Save feature matrix
        feat_name = raw_path.stem + "_features.parquet"
        feat_path = FEATURES_DIR / feat_name
        features.to_parquet(feat_path)

        # Save target vector
        target_name = raw_path.stem + "_target.parquet"
        target_path = FEATURES_DIR / target_name
        target.to_frame().to_parquet(target_path)

        print(f"    ✓ X: {features.shape[0]} rows × {features.shape[1]} features → {feat_name}")
        print(f"    ✓ y: {target.sum()}/{len(target)} positive → {target_name}")

    print("\n✅ Feature engineering complete.\n")


if __name__ == "__main__":
    main()
