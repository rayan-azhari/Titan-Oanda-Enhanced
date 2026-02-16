"""validate_data.py — Data quality validation for raw OHLCV Parquet files.

Checks for weekend gaps, missing candles, duplicate timestamps,
and price spike outliers. Should be run BEFORE build_ml_features.py.

Research Gap Fix: No data quality validation was specified in the
original research document.
"""

import sys
from pathlib import Path

import pandas as pd



# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------


def check_duplicates(df: pd.DataFrame, name: str) -> int:
    """Check for duplicate timestamps.

    Args:
        df: OHLCV DataFrame with a 'timestamp' column.
        name: Dataset identifier for reporting.

    Returns:
        Number of duplicates found.
    """
    dupes = df["timestamp"].duplicated().sum()
    if dupes > 0:
        print(f"  ⚠️  {name}: {dupes} duplicate timestamps found")
    else:
        print(f"  ✓ {name}: No duplicates")
    return dupes


def check_gaps(df: pd.DataFrame, name: str, expected_freq: str = "5min") -> int:
    """Check for missing candles (excluding weekends).

    Args:
        df: OHLCV DataFrame with a 'timestamp' column.
        name: Dataset identifier for reporting.
        expected_freq: Expected candle frequency (e.g., "5min", "1h").

    Returns:
        Number of unexpected gaps found.
    """
    ts = pd.to_datetime(df["timestamp"]).sort_values()

    # Generate expected range excluding weekends (Sat=5, Sun=6)
    full_range = pd.date_range(start=ts.min(), end=ts.max(), freq=expected_freq)
    full_range = full_range[full_range.dayofweek < 5]  # Remove Sat/Sun

    # Also remove Friday 22:00+ to Sunday 22:00 (forex market closed)
    actual_set = set(ts)
    missing = [t for t in full_range if t not in actual_set]

    # Filter out known market-closed hours (rough filter)
    # Forex: closed Fri ~22:00 UTC to Sun ~22:00 UTC
    real_missing = []
    for t in missing:
        # Skip Friday after 21:00 UTC and all of Saturday
        if t.dayofweek == 4 and t.hour >= 22:
            continue
        real_missing.append(t)

    n_missing = len(real_missing)
    if n_missing > 0:
        print(f"  ⚠️  {name}: {n_missing} missing candles (excl. weekends)")
        if n_missing <= 10:
            for t in real_missing[:10]:
                print(f"       → {t}")
    else:
        print(f"  ✓ {name}: No gaps detected")
    return n_missing


def check_outliers(df: pd.DataFrame, name: str, z_threshold: float = 5.0) -> int:
    """Check for price spike outliers using z-score.

    A z-score above the threshold on return_1 indicates a suspicious
    price spike that may be a broker-side error.

    Args:
        df: OHLCV DataFrame with a 'close' column.
        name: Dataset identifier for reporting.
        z_threshold: Z-score threshold for flagging outliers.

    Returns:
        Number of outliers detected.
    """
    close = df["close"].astype(float)
    returns = close.pct_change().dropna()

    mean = returns.mean()
    std = returns.std()
    if std == 0:
        print(f"  ✓ {name}: No variance in returns (constant price)")
        return 0

    z_scores = ((returns - mean) / std).abs()
    outliers = z_scores[z_scores > z_threshold]

    if len(outliers) > 0:
        print(f"  ⚠️  {name}: {len(outliers)} price spikes (|z| > {z_threshold})")
        for idx, z in outliers.head(5).items():
            print(f"       → {idx}: z={z:.2f}, return={returns.loc[idx]:.6f}")
    else:
        print(f"  ✓ {name}: No outlier spikes")
    return len(outliers)


def check_negative_volume(df: pd.DataFrame, name: str) -> int:
    """Check for zero or negative volume candles.

    Args:
        df: OHLCV DataFrame with a 'volume' column.
        name: Dataset identifier for reporting.

    Returns:
        Number of zero/negative volume rows.
    """
    bad = (df["volume"] <= 0).sum()
    if bad > 0:
        print(f"  ⚠️  {name}: {bad} candles with zero/negative volume")
    else:
        print(f"  ✓ {name}: All volumes positive")
    return bad


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------



