"""features.py — Shared feature engineering logic for ML training and live inference.

This module ensures that the features calculated during training (run_ml_strategy.py)
are IDENTICAL to the features calculated during live trading (ml_strategy.py).
"""

import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"

# ─────────────────────────────────────────────────────────────────────
# Feature Config (from VBT Feature Selection)
# ─────────────────────────────────────────────────────────────────────

# Default params used when config/features.toml is missing.
_DEFAULT_CFG = {
    "trend": {
        "sma_periods": [20, 50],
        "ema_periods": [12, 26],
        "macd": {"fast": 12, "slow": 26, "signal": 9},
    },
    "momentum": {
        "rsi": {"window": 14, "entry": 30},
        "stochastic": {"k": 14, "d": 3},
    },
    "volatility": {
        "bollinger": {"window": 20, "std_dev": 2.0},
        "adx": {"period": 14, "threshold": 25},
    },
    "mtf_confluence": {
        "higher_tfs": ["D", "W"],
        "D": {
            "sma_fast": 10,
            "sma_slow": 30,
            "rsi_period": 14,
            "rsi_threshold": 50,
        },
        "W": {
            "sma_fast": 5,
            "sma_slow": 13,
            "rsi_period": 14,
            "rsi_threshold": 50,
        },
    },
}


def load_feature_config(
    logger=None,
) -> dict:
    """Load tuned feature parameters from config/features.toml.

    Falls back to hardcoded defaults if the file is missing,
    ensuring backward compatibility.
    """
    cfg_path = CONFIG_DIR / "features.toml"
    if cfg_path.exists():
        with open(cfg_path, "rb") as f:
            cfg = tomllib.load(f)
        gen = cfg.get("selection", {}).get("generated_at", "unknown")
        msg = f"  📋 Loaded tuned features from features.toml ({gen})"
        if logger:
            logger.info(msg)
        else:
            print(msg)
        return cfg

    msg = "  ⚠ config/features.toml not found — using defaults."
    if logger:
        logger.warning(msg)
    else:
        print(msg)
    return _DEFAULT_CFG


# ─────────────────────────────────────────────────────────────────────
# Technical Indicators
# ─────────────────────────────────────────────────────────────────────


def sma(s: pd.Series, p: int) -> pd.Series:
    return s.rolling(p).mean()


def ema(s: pd.Series, p: int) -> pd.Series:
    return s.ewm(span=p, adjust=False).mean()


def rsi(s: pd.Series, p: int = 14) -> pd.Series:
    d = s.diff()
    g = d.where(d > 0, 0.0).rolling(p).mean()
    loss = (-d.where(d < 0, 0.0)).rolling(p).mean()
    return 100 - (100 / (1 + g / loss))


def atr(df: pd.DataFrame, p: int = 14) -> pd.Series:
    h, low, c = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [h - low, (h - c.shift(1)).abs(), (low - c.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(p).mean()


def macd_hist(s: pd.Series, fast=12, slow=26, sig=9) -> pd.Series:
    m = ema(s, fast) - ema(s, slow)
    return m - ema(m, sig)


def bollinger_bw(s: pd.Series, p: int = 20) -> pd.Series:
    mid = sma(s, p)
    std = s.rolling(p).std()
    return (2 * 2 * std) / mid


def stochastic(
    df: pd.DataFrame, k_period: int = 14, d_period: int = 3
) -> tuple[pd.Series, pd.Series]:
    """Stochastic oscillator %K and %D."""
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min)
    d = k.rolling(d_period).mean()
    return k, d


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index (simplified)."""
    h, low, c = df["high"], df["low"], df["close"]
    plus_dm = h.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.concat(
        [h - low, (h - c.shift(1)).abs(), (low - c.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr_v = tr.rolling(period).mean()
    plus_di = 100 * plus_dm.rolling(period).mean() / atr_v
    minus_di = 100 * minus_dm.rolling(period).mean() / atr_v
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.rolling(period).mean()


# ─────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────


def build_features(
    df: pd.DataFrame,
    context_data: dict[str, pd.DataFrame] | None = None,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """Build feature matrix using tuned parameters from config.

    If cfg is None, loads config/features.toml automatically.
    Falls back to hardcoded defaults if no config exists.
    """
    if cfg is None:
        cfg = load_feature_config()

    if context_data is None:
        context_data = {}

    close = df["close"]
    feats = pd.DataFrame(index=df.index)

    # ── Config sections (with defaults) ──
    trend = cfg.get("trend", _DEFAULT_CFG["trend"])
    momentum = cfg.get("momentum", _DEFAULT_CFG["momentum"])
    volatility = cfg.get("volatility", _DEFAULT_CFG["volatility"])
    mtf_cfg = cfg.get("mtf_confluence", _DEFAULT_CFG["mtf_confluence"])

    # ── Lagged returns ──
    for lag in [1, 2, 3, 5, 10, 20]:
        feats[f"ret_{lag}"] = close.pct_change(lag)

    # ── Trend (tuned) ──
    sma_periods = trend.get("sma_periods", [20, 50])
    for p in sma_periods:
        feats[f"sma_{p}"] = sma(close, p) / close - 1

    ema_periods = trend.get("ema_periods", [12, 26])
    feats["ema_slope"] = ema(close, ema_periods[0]).diff() / close

    macd_cfg = trend.get("macd", {"fast": 12, "slow": 26, "signal": 9})
    if macd_cfg is True:
        macd_cfg = {"fast": 12, "slow": 26, "signal": 9}
    if isinstance(macd_cfg, dict):
        feats["macd_hist"] = macd_hist(
            close,
            fast=macd_cfg.get("fast", 12),
            slow=macd_cfg.get("slow", 26),
            sig=macd_cfg.get("signal", 9),
        )

    # MA crossover (fast vs slow from SMA tuning)
    if len(sma_periods) >= 2:
        feats["ma_cross"] = (sma(close, sma_periods[0]) > sma(close, sma_periods[1])).astype(float)

    # ── Momentum (tuned) ──
    rsi_cfg = momentum.get("rsi", {"window": 14})
    if rsi_cfg is True:
        rsi_cfg = {"window": 14}

    if isinstance(rsi_cfg, dict):
        rsi_win = rsi_cfg.get("window", 14)
        feats["rsi"] = rsi(close, rsi_win)
        # Also include a shorter RSI for signal diversity
        feats["rsi_fast"] = rsi(close, max(5, rsi_win // 2))

    stoch_cfg = momentum.get("stochastic", {"k": 14, "d": 3})
    if stoch_cfg is True:
        stoch_cfg = {"k": 14, "d": 3}
    if isinstance(stoch_cfg, dict):
        stoch_k, stoch_d = stochastic(
            df,
            k_period=stoch_cfg.get("k", 14),
            d_period=stoch_cfg.get("d", 3),
        )
        feats["stoch_k"] = stoch_k
        feats["stoch_d"] = stoch_d

    # ── Volatility (tuned) ──
    feats["atr_14"] = atr(df, 14) / close

    bb_cfg = volatility.get("bollinger", {"window": 20, "std_dev": 2.0})
    if bb_cfg is True:
        bb_cfg = {"window": 20, "std_dev": 2.0}

    if isinstance(bb_cfg, dict):
        feats["boll_bw"] = bollinger_bw(close, p=bb_cfg.get("window", 20))
    feats["close_std_20"] = close.rolling(20).std() / close

    adx_cfg = volatility.get("adx", {"period": 14})
    if adx_cfg is True:
        adx_cfg = {"period": 14}

    if isinstance(adx_cfg, dict):
        feats["adx"] = adx(df, period=adx_cfg.get("period", 14))

    # ── Volume ──
    if "volume" in df.columns:
        feats["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
        feats["vol_change"] = df["volume"].pct_change()
        feats["vol_rsi"] = rsi(df["volume"], 14)

    # ── Price action ──
    feats["range_pct"] = (df["high"] - df["low"]) / close

    # ── MTF Confluence (tuned) ──
    for gran, context_df in context_data.items():
        prefix = gran.lower()
        ctx_close = context_df["close"]

        # Load tuned params for this TF (or defaults)
        tf_cfg = mtf_cfg.get(gran, {})
        sf = tf_cfg.get("sma_fast", 10 if gran == "D" else 5)
        ss = tf_cfg.get("sma_slow", 30 if gran == "D" else 13)
        rp = tf_cfg.get("rsi_period", 14)
        rt = tf_cfg.get("rsi_threshold", 50)

        fast_ma = sma(ctx_close, sf)
        slow_ma = sma(ctx_close, ss)
        r = rsi(ctx_close, rp)

        bias = pd.Series(0.0, index=ctx_close.index)
        bias += np.where(fast_ma > slow_ma, 0.5, -0.5)
        bias += np.where(r > rt, 0.5, -0.5)

        bias_aligned = bias.reindex(feats.index, method="ffill")
        feats[f"{prefix}_bias"] = bias_aligned

        trend_str = ((fast_ma - slow_ma) / ctx_close).reindex(feats.index, method="ffill")
        feats[f"{prefix}_trend_str"] = trend_str

        r_aligned = r.reindex(feats.index, method="ffill")
        feats[f"{prefix}_rsi"] = r_aligned

    return feats
