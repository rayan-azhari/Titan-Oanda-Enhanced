"""run_feature_selection.py — VBT-based indicator tuning & feature selection.

Sweeps 7 indicator families across parameter ranges using VectorBT,
scores each by IS/OOS Sharpe stability, tunes MTF confluence filters,
and writes optimal parameters to config/features.toml for the ML pipeline.

Directive: Alpha Research Loop (VectorBT).md
"""

import json
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    import vectorbt as vbt
except ImportError:
    print("ERROR: vectorbt is not installed. Run `uv sync` first.")
    sys.exit(1)

from titan.models.spread import build_spread_series  # noqa: E402

# ─────────────────────────────────────────────────────────────────────
# Granularity → pandas freq mapping
# ─────────────────────────────────────────────────────────────────────

GRAN_TO_FREQ: dict[str, str] = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D": "1D",
    "W": "1W",
}


# ─────────────────────────────────────────────────────────────────────
# Data Loading (reused from run_vbt_optimisation.py)
# ─────────────────────────────────────────────────────────────────────


def load_instruments_config() -> dict:
    """Load the instruments configuration from config/instruments.toml."""
    config_path = CONFIG_DIR / "instruments.toml"
    if not config_path.exists():
        print(f"ERROR: {config_path} not found.")
        sys.exit(1)
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def load_data(pair: str, granularity: str) -> pd.DataFrame | None:
    """Load Parquet OHLCV data for a given pair and granularity."""
    path = RAW_DATA_DIR / f"{pair}_{granularity}.parquet"
    if not path.exists():
        print(f"  ⚠ {path.name} not found — skipping.")
        return None
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    if "volume" in df.columns:
        df["volume"] = df["volume"].astype(float)
    freq = GRAN_TO_FREQ.get(granularity)
    if freq:
        df = df.asfreq(freq, method="pad")
    return df


def split_data(df: pd.DataFrame, ratio: float = 0.70) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into in-sample (70%) and out-of-sample (30%)."""
    split_idx = int(len(df) * ratio)
    return df.iloc[:split_idx], df.iloc[split_idx:]


# ─────────────────────────────────────────────────────────────────────
# Technical Indicator Helpers (for non-VBT-native indicators)
# ─────────────────────────────────────────────────────────────────────


def calc_sma(s: pd.Series, p: int) -> pd.Series:
    """Simple Moving Average."""
    return s.rolling(p).mean()


def calc_ema(s: pd.Series, p: int) -> pd.Series:
    """Exponential Moving Average."""
    return s.ewm(span=p, adjust=False).mean()


def calc_rsi(s: pd.Series, p: int = 14) -> pd.Series:
    """Relative Strength Index."""
    d = s.diff()
    gain = d.where(d > 0, 0.0).rolling(p).mean()
    loss = (-d.where(d < 0, 0.0)).rolling(p).mean()
    return 100 - (100 / (1 + gain / loss))


def calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
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
# Individual Indicator Sweep Functions
# ─────────────────────────────────────────────────────────────────────


def sweep_rsi(close: pd.Series, fees: float = 0.0003) -> list[dict]:
    """Sweep RSI window × entry threshold."""
    windows = list(range(5, 31))
    thresholds = list(range(15, 41))

    rsi_ind = vbt.RSI.run(close, window=windows, param_product=True)
    entries = rsi_ind.rsi_crossed_below(thresholds)
    exits = rsi_ind.rsi_crossed_above(100 - np.array(thresholds))

    pf = vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=exits,
        init_cash=10_000,
        fees=fees,
    )
    sharpe = pf.sharpe_ratio()

    results = []
    for w in windows:
        for t in thresholds:
            try:
                s = float(sharpe.loc[(w, t)])
            except (KeyError, TypeError):
                continue
            if np.isfinite(s):
                results.append(
                    {
                        "indicator": "RSI",
                        "params": {"window": w, "entry": t},
                        "sharpe": round(s, 4),
                    }
                )
    return results


def sweep_sma_cross(close: pd.Series, fees: float = 0.0003) -> list[dict]:
    """Sweep SMA fast × slow crossover."""
    fasts = list(range(5, 31, 2))
    slows = list(range(20, 101, 5))

    results = []
    for fast in fasts:
        for slow in slows:
            if fast >= slow:
                continue
            fast_ma = calc_sma(close, fast)
            slow_ma = calc_sma(close, slow)
            entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
            exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
            pf = vbt.Portfolio.from_signals(
                close,
                entries=entries,
                exits=exits,
                init_cash=10_000,
                fees=fees,
            )
            s = float(pf.sharpe_ratio())
            if np.isfinite(s):
                results.append(
                    {
                        "indicator": "SMA_Cross",
                        "params": {"fast": fast, "slow": slow},
                        "sharpe": round(s, 4),
                    }
                )
    return results


def sweep_ema_cross(close: pd.Series, fees: float = 0.0003) -> list[dict]:
    """Sweep EMA fast × slow crossover."""
    fasts = list(range(5, 26, 2))
    slows = list(range(15, 61, 3))

    results = []
    for fast in fasts:
        for slow in slows:
            if fast >= slow:
                continue
            fast_ma = calc_ema(close, fast)
            slow_ma = calc_ema(close, slow)
            entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
            exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
            pf = vbt.Portfolio.from_signals(
                close,
                entries=entries,
                exits=exits,
                init_cash=10_000,
                fees=fees,
            )
            s = float(pf.sharpe_ratio())
            if np.isfinite(s):
                results.append(
                    {
                        "indicator": "EMA_Cross",
                        "params": {"fast": fast, "slow": slow},
                        "sharpe": round(s, 4),
                    }
                )
    return results


def sweep_macd(close: pd.Series, fees: float = 0.0003) -> list[dict]:
    """Sweep MACD fast/slow/signal."""
    fast_vals = list(range(8, 17, 2))
    slow_vals = list(range(20, 31, 2))
    sig_vals = list(range(7, 13))

    results = []
    for fast in fast_vals:
        for slow in slow_vals:
            if fast >= slow:
                continue
            for sig in sig_vals:
                macd_line = calc_ema(close, fast) - calc_ema(close, slow)
                signal_line = calc_ema(macd_line, sig)
                hist = macd_line - signal_line

                entries = (hist > 0) & (hist.shift(1) <= 0)
                exits = (hist < 0) & (hist.shift(1) >= 0)

                pf = vbt.Portfolio.from_signals(
                    close,
                    entries=entries,
                    exits=exits,
                    init_cash=10_000,
                    fees=fees,
                )
                s = float(pf.sharpe_ratio())
                if np.isfinite(s):
                    results.append(
                        {
                            "indicator": "MACD",
                            "params": {
                                "fast": fast,
                                "slow": slow,
                                "signal": sig,
                            },
                            "sharpe": round(s, 4),
                        }
                    )
    return results


def sweep_bollinger(close: pd.Series, fees: float = 0.0003) -> list[dict]:
    """Sweep Bollinger Band window × std_dev."""
    windows = list(range(10, 31, 2))
    std_devs = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]

    results = []
    for w in windows:
        mid = calc_sma(close, w)
        std = close.rolling(w).std()
        for sd in std_devs:
            upper = mid + sd * std
            lower = mid - sd * std

            entries = close < lower  # Buy at lower band
            exits = close > upper  # Sell at upper band

            pf = vbt.Portfolio.from_signals(
                close,
                entries=entries,
                exits=exits,
                init_cash=10_000,
                fees=fees,
            )
            s = float(pf.sharpe_ratio())
            if np.isfinite(s):
                results.append(
                    {
                        "indicator": "Bollinger",
                        "params": {"window": w, "std_dev": sd},
                        "sharpe": round(s, 4),
                    }
                )
    return results


def sweep_stochastic(df: pd.DataFrame, fees: float = 0.0003) -> list[dict]:
    """Sweep Stochastic %K period × %D period."""
    k_periods = list(range(5, 22, 2))
    d_periods = [2, 3, 4, 5]
    close = df["close"]

    results = []
    for k in k_periods:
        low_min = df["low"].rolling(k).min()
        high_max = df["high"].rolling(k).max()
        pct_k = 100 * (close - low_min) / (high_max - low_min)

        for d in d_periods:
            pct_d = pct_k.rolling(d).mean()

            entries = (pct_k < 20) & (pct_k > pct_d)
            exits = (pct_k > 80) & (pct_k < pct_d)

            pf = vbt.Portfolio.from_signals(
                close,
                entries=entries,
                exits=exits,
                init_cash=10_000,
                fees=fees,
            )
            s = float(pf.sharpe_ratio())
            if np.isfinite(s):
                results.append(
                    {
                        "indicator": "Stochastic",
                        "params": {"k_period": k, "d_period": d},
                        "sharpe": round(s, 4),
                    }
                )
    return results


def sweep_adx_filter(
    df: pd.DataFrame,
    base_entries: pd.Series,
    base_exits: pd.Series,
    fees: float = 0.0003,
) -> list[dict]:
    """Sweep ADX as a trend filter on top of a base strategy."""
    periods = list(range(10, 26, 2))
    thresholds = list(range(20, 31, 2))
    close = df["close"]

    results = []
    for period in periods:
        adx_val = calc_adx(df, period)
        for threshold in thresholds:
            trend_mask = adx_val > threshold
            filtered_entries = base_entries & trend_mask
            filtered_exits = base_exits  # exits are unrestricted

            pf = vbt.Portfolio.from_signals(
                close,
                entries=filtered_entries,
                exits=filtered_exits,
                init_cash=10_000,
                fees=fees,
            )
            s = float(pf.sharpe_ratio())
            if np.isfinite(s):
                results.append(
                    {
                        "indicator": "ADX_Filter",
                        "params": {
                            "period": period,
                            "threshold": threshold,
                        },
                        "sharpe": round(s, 4),
                    }
                )
    return results


# ─────────────────────────────────────────────────────────────────────
# Multi-Timeframe Confluence Sweep
# ─────────────────────────────────────────────────────────────────────


def sweep_mtf_confluence(
    base_df: pd.DataFrame,
    base_entries: pd.Series,
    base_exits: pd.Series,
    pair: str,
    base_gran: str,
    fees: float = 0.0003,
) -> list[dict]:
    """Sweep MTF bias filter parameters on higher timeframes.

    Tests which higher-TF + SMA/RSI params produce the best
    trade filter when layered on top of a base strategy.
    """
    # Determine which higher TFs to test
    tf_order = ["H1", "H4", "D", "W"]
    base_idx = tf_order.index(base_gran) if base_gran in tf_order else 1
    higher_tfs = tf_order[base_idx + 1 :]

    sma_fasts = [5, 8, 10, 13, 15, 20]
    sma_slows = [13, 20, 26, 30, 40, 50, 60]
    rsi_periods = [7, 9, 14, 21]
    rsi_thresholds = [40, 45, 50, 55, 60]

    close = base_df["close"]
    base_index = base_df.index
    results = []

    for tf in higher_tfs:
        htf_df = load_data(pair, tf)
        if htf_df is None or len(htf_df) < 60:
            print(f"    ⚠ {tf} data insufficient, skipping.")
            continue

        htf_close = htf_df["close"]

        for sf in sma_fasts:
            for ss in sma_slows:
                if sf >= ss:
                    continue
                fast_ma = calc_sma(htf_close, sf)
                slow_ma = calc_sma(htf_close, ss)
                trend_bias = (fast_ma > slow_ma).astype(float)

                for rp in rsi_periods:
                    htf_rsi = calc_rsi(htf_close, rp)

                    for rt in rsi_thresholds:
                        rsi_bias = (htf_rsi > rt).astype(float)

                        # Combined bias: both must agree for bullish
                        bullish = (trend_bias == 1) & (rsi_bias == 1)

                        # Align to base timeframe
                        bullish_aligned = bullish.reindex(base_index, method="ffill").fillna(False)

                        # Filter: only enter when higher TF is bullish
                        filtered_entries = base_entries & bullish_aligned

                        pf = vbt.Portfolio.from_signals(
                            close,
                            entries=filtered_entries,
                            exits=base_exits,
                            init_cash=10_000,
                            fees=fees,
                        )
                        s = float(pf.sharpe_ratio())
                        if np.isfinite(s):
                            results.append(
                                {
                                    "indicator": f"MTF_{tf}",
                                    "params": {
                                        "timeframe": tf,
                                        "sma_fast": sf,
                                        "sma_slow": ss,
                                        "rsi_period": rp,
                                        "rsi_threshold": rt,
                                    },
                                    "sharpe": round(s, 4),
                                }
                            )

    return results


# ─────────────────────────────────────────────────────────────────────
# Scoring & Selection
# ─────────────────────────────────────────────────────────────────────


def score_indicator(is_results: list[dict], oos_results: list[dict]) -> list[dict]:
    """Match IS/OOS results and compute stability score.

    Stability = min(IS, OOS) / max(IS, OOS).
    Only retains combos where both IS and OOS Sharpe > 0.
    """

    # Build lookup: param_key → sharpe
    def param_key(r: dict) -> str:
        return f"{r['indicator']}|{json.dumps(r['params'], sort_keys=True)}"

    is_map = {param_key(r): r["sharpe"] for r in is_results}
    oos_map = {param_key(r): r["sharpe"] for r in oos_results}

    scored = []
    for key in is_map:
        if key not in oos_map:
            continue
        is_s = is_map[key]
        oos_s = oos_map[key]
        if is_s <= 0 or oos_s <= 0:
            continue

        max_s = max(is_s, oos_s)
        stability = min(is_s, oos_s) / max_s if max_s > 0 else 0

        # Parse back
        indicator, params_json = key.split("|", 1)
        params = json.loads(params_json)

        scored.append(
            {
                "indicator": indicator,
                "params": params,
                "sharpe_is": is_s,
                "sharpe_oos": oos_s,
                "stability": round(stability, 4),
                "score": round(oos_s * stability, 4),
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def select_best_per_indicator(
    scoreboard: list[dict], min_oos_sharpe: float = 0.3
) -> dict[str, dict]:
    """Select the best parameter set for each indicator family."""
    best = {}
    for entry in scoreboard:
        ind = entry["indicator"]
        if entry["sharpe_oos"] < min_oos_sharpe:
            continue
        if ind not in best:
            best[ind] = entry
    return best


# ─────────────────────────────────────────────────────────────────────
# Config Output
# ─────────────────────────────────────────────────────────────────────


def write_features_toml(
    best: dict[str, dict],
    pair: str,
    granularity: str,
    scoreboard: list[dict],
) -> Path:
    """Write tuned parameters to config/features.toml."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out_path = CONFIG_DIR / "features.toml"

    lines = [
        "# ──────────────────────────────────────────────────────────",
        "# features.toml — Auto-generated by run_feature_selection.py",
        f"# Generated: {now}",
        "# Re-run the sweep to update. Manual edits will be overwritten.",
        "# ──────────────────────────────────────────────────────────",
        "",
        "[selection]",
        f'generated_at = "{now}"',
        f'pair = "{pair}"',
        f'granularity = "{granularity}"',
        "min_sharpe_oos = 0.3",
        "",
    ]

    # ── Trend indicators ──
    lines.append("[trend]")
    if "SMA_Cross" in best:
        p = best["SMA_Cross"]["params"]
        s = best["SMA_Cross"]["sharpe_oos"]
        lines.append(f"sma_periods = [{p['fast']}, {p['slow']}]")
        lines.append(f"# VBT: SMA cross OOS Sharpe {s}")
    else:
        lines.append("sma_periods = [20, 50]  # default (not tuned)")

    if "EMA_Cross" in best:
        p = best["EMA_Cross"]["params"]
        s = best["EMA_Cross"]["sharpe_oos"]
        lines.append(f"ema_periods = [{p['fast']}, {p['slow']}]")
        lines.append(f"# VBT: EMA cross OOS Sharpe {s}")
    else:
        lines.append("ema_periods = [12, 26]  # default")

    if "MACD" in best:
        p = best["MACD"]["params"]
        s = best["MACD"]["sharpe_oos"]
        fast, slow, sig = p["fast"], p["slow"], p["signal"]
        lines.append(f"macd = {{ fast = {fast}, slow = {slow}, signal = {sig} }}")
        lines.append(f"# VBT: MACD OOS Sharpe {s}")
    else:
        lines.append("macd = { fast = 12, slow = 26, signal = 9 }")
    lines.append("")

    # ── Momentum indicators ──
    lines.append("[momentum]")
    if "RSI" in best:
        p = best["RSI"]["params"]
        s = best["RSI"]["sharpe_oos"]
        lines.append(f"rsi = {{ window = {p['window']}, entry = {p['entry']} }}")
        lines.append(f"# VBT: RSI OOS Sharpe {s}")
    else:
        lines.append("rsi = { window = 14, entry = 30 }  # default")

    if "Stochastic" in best:
        p = best["Stochastic"]["params"]
        s = best["Stochastic"]["sharpe_oos"]
        lines.append(f"stochastic = {{ k = {p['k_period']}, d = {p['d_period']} }}")
        lines.append(f"# VBT: Stochastic OOS Sharpe {s}")
    else:
        lines.append("stochastic = { k = 14, d = 3 }  # default")
    lines.append("")

    # ── Volatility indicators ──
    lines.append("[volatility]")
    if "Bollinger" in best:
        p = best["Bollinger"]["params"]
        s = best["Bollinger"]["sharpe_oos"]
        lines.append(f"bollinger = {{ window = {p['window']}, std_dev = {p['std_dev']} }}")
        lines.append(f"# VBT: Bollinger OOS Sharpe {s}")
    else:
        lines.append("bollinger = { window = 20, std_dev = 2.0 }  # default")

    if "ADX_Filter" in best:
        p = best["ADX_Filter"]["params"]
        s = best["ADX_Filter"]["sharpe_oos"]
        lines.append(f"adx = {{ period = {p['period']}, threshold = {p['threshold']} }}")
        lines.append(f"# VBT: ADX filter OOS Sharpe {s}")
    else:
        lines.append("adx = { period = 14, threshold = 25 }  # default")
    lines.append("")

    # ── MTF Confluence ──
    mtf_keys = [k for k in best if k.startswith("MTF_")]
    if mtf_keys:
        tfs_used = [best[k]["params"]["timeframe"] for k in mtf_keys]
        tf_str = ", ".join(f'"{tf}"' for tf in tfs_used)
        lines.append("[mtf_confluence]")
        lines.append(f"higher_tfs = [{tf_str}]")
        lines.append("")

        for mk in mtf_keys:
            p = best[mk]["params"]
            s = best[mk]["sharpe_oos"]
            tf = p["timeframe"]
            lines.append(f"[mtf_confluence.{tf}]")
            lines.append(f"sma_fast = {p['sma_fast']}")
            lines.append(f"sma_slow = {p['sma_slow']}")
            lines.append(f"rsi_period = {p['rsi_period']}")
            lines.append(f"rsi_threshold = {p['rsi_threshold']}")
            lines.append(f"# VBT: MTF {tf} filtered OOS Sharpe {s}")
            lines.append("")
    else:
        lines.append("[mtf_confluence]")
        lines.append('higher_tfs = ["D", "W"]  # default')
        lines.append("")
        lines.append("[mtf_confluence.D]")
        lines.append("sma_fast = 10")
        lines.append("sma_slow = 30")
        lines.append("rsi_period = 14")
        lines.append("rsi_threshold = 50")
        lines.append("")
        lines.append("[mtf_confluence.W]")
        lines.append("sma_fast = 5")
        lines.append("sma_slow = 13")
        lines.append("rsi_period = 14")
        lines.append("rsi_threshold = 50")
        lines.append("")

    # ── Scoring summary ──
    lines.append("[scoring]")
    if scoreboard:
        top = scoreboard[0]
        lines.append(f'best_indicator = "{top["indicator"]}"')
        lines.append(f"best_sharpe_oos = {top['sharpe_oos']}")
        lines.append(f"best_stability = {top['stability']}")
    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\n  ✅ Wrote tuned parameters → {out_path.name}")
    return out_path


# ─────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────


def run_sweep_for_pair(
    pair: str,
    granularity: str,
) -> list[dict]:
    """Run the full indicator sweep for one pair."""
    print(f"\n{'=' * 60}")
    print(f"🔬 Feature Selection Sweep: {pair} ({granularity})")
    print(f"{'=' * 60}")

    df = load_data(pair, granularity)
    if df is None:
        return []

    is_df, oos_df = split_data(df)
    print(f"  IS: {len(is_df)} bars ({is_df.index.min()} → {is_df.index.max()})")
    print(f"  OOS: {len(oos_df)} bars ({oos_df.index.min()} → {oos_df.index.max()})")

    # Calculate spread for transaction costs
    try:
        spread_series = build_spread_series(df, pair)
        fees = float(spread_series.mean())
    except Exception:
        fees = 0.0003  # Default 3 pips
    print(f"  📊 Transaction cost: {fees * 10000:.1f} pips\n")

    # ── Run all 6 indicator sweeps on IS and OOS ──
    sweep_funcs = [
        ("RSI", lambda d: sweep_rsi(d["close"], fees)),
        ("SMA Cross", lambda d: sweep_sma_cross(d["close"], fees)),
        ("EMA Cross", lambda d: sweep_ema_cross(d["close"], fees)),
        ("MACD", lambda d: sweep_macd(d["close"], fees)),
        ("Bollinger", lambda d: sweep_bollinger(d["close"], fees)),
        ("Stochastic", lambda d: sweep_stochastic(d, fees)),
    ]

    all_is_results = []
    all_oos_results = []

    for name, fn in sweep_funcs:
        print(f"  ▶ Sweeping {name}...")
        is_r = fn(is_df)
        oos_r = fn(oos_df)
        all_is_results.extend(is_r)
        all_oos_results.extend(oos_r)
        print(f"    {len(is_r)} IS combos, {len(oos_r)} OOS combos")

    # ── Score and select best base indicator ──
    scoreboard = score_indicator(all_is_results, all_oos_results)
    best = select_best_per_indicator(scoreboard)

    print("\n  🏆 Best per indicator (OOS Sharpe ≥ 0.3):")
    for ind, entry in best.items():
        print(
            f"     {ind}: "
            f"IS={entry['sharpe_is']:.3f} "
            f"OOS={entry['sharpe_oos']:.3f} "
            f"Stability={entry['stability']:.3f} "
            f"Params={entry['params']}"
        )

    # ── ADX as overlay filter on best base ──
    if best:
        print("\n  ▶ Sweeping ADX filter on best base strategy...")
        top_ind = list(best.values())[0]
        # Reconstruct base entries/exits for ADX sweep
        base_entries, base_exits = _reconstruct_signals(is_df, top_ind)
        if base_entries is not None:
            is_adx = sweep_adx_filter(is_df, base_entries, base_exits, fees)
            base_entries_oos, base_exits_oos = _reconstruct_signals(oos_df, top_ind)
            oos_adx = sweep_adx_filter(oos_df, base_entries_oos, base_exits_oos, fees)
            adx_scored = score_indicator(is_adx, oos_adx)
            adx_best = select_best_per_indicator(adx_scored)
            best.update(adx_best)
            print(f"    {len(is_adx)} IS combos, {len(oos_adx)} OOS combos")

    # ── MTF Confluence Sweep ──
    if best:
        print("\n  ▶ Sweeping MTF Confluence filters...")
        top_ind = list(best.values())[0]
        base_entries_full, base_exits_full = _reconstruct_signals(df, top_ind)
        if base_entries_full is not None:
            is_entries, _ = _reconstruct_signals(is_df, top_ind)
            is_exits = base_exits_full.reindex(is_df.index).fillna(False)
            is_mtf = sweep_mtf_confluence(is_df, is_entries, is_exits, pair, granularity, fees)

            oos_entries, oos_exits = _reconstruct_signals(oos_df, top_ind)
            oos_mtf = sweep_mtf_confluence(oos_df, oos_entries, oos_exits, pair, granularity, fees)

            mtf_scored = score_indicator(is_mtf, oos_mtf)
            mtf_best = select_best_per_indicator(mtf_scored)
            best.update(mtf_best)

            if mtf_best:
                print("    🏆 Best MTF configs:")
                for ind, entry in mtf_best.items():
                    print(f"       {ind}: OOS={entry['sharpe_oos']:.3f} Params={entry['params']}")
            else:
                print("    ⚠ No MTF config passed threshold.")

    # ── Write outputs ──
    write_features_toml(best, pair, granularity, scoreboard)

    # Save full scoreboard as JSON
    scoreboard_path = REPORTS_DIR / f"feature_scoreboard_{pair}.json"
    with open(scoreboard_path, "w") as f:
        json.dump(scoreboard[:100], f, indent=2)  # Top 100
    print(f"  📋 Scoreboard → {scoreboard_path.name}")

    return scoreboard


def _reconstruct_signals(
    df: pd.DataFrame, best_entry: dict
) -> tuple[pd.Series | None, pd.Series | None]:
    """Reconstruct entry/exit signals from a best-indicator entry."""
    ind = best_entry["indicator"]
    p = best_entry["params"]
    close = df["close"]

    if ind == "RSI":
        r = calc_rsi(close, p["window"])
        entries = r < p["entry"]
        exits = r > (100 - p["entry"])
    elif ind == "SMA_Cross":
        fast_ma = calc_sma(close, p["fast"])
        slow_ma = calc_sma(close, p["slow"])
        entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
    elif ind == "EMA_Cross":
        fast_ma = calc_ema(close, p["fast"])
        slow_ma = calc_ema(close, p["slow"])
        entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
    elif ind == "MACD":
        macd_line = calc_ema(close, p["fast"]) - calc_ema(close, p["slow"])
        signal_line = calc_ema(macd_line, p["signal"])
        hist = macd_line - signal_line
        entries = (hist > 0) & (hist.shift(1) <= 0)
        exits = (hist < 0) & (hist.shift(1) >= 0)
    elif ind == "Bollinger":
        mid = calc_sma(close, p["window"])
        std = close.rolling(p["window"]).std()
        entries = close < (mid - p["std_dev"] * std)
        exits = close > (mid + p["std_dev"] * std)
    elif ind == "Stochastic":
        low_min = df["low"].rolling(p["k_period"]).min()
        high_max = df["high"].rolling(p["k_period"]).max()
        pct_k = 100 * (close - low_min) / (high_max - low_min)
        pct_d = pct_k.rolling(p["d_period"]).mean()
        entries = (pct_k < 20) & (pct_k > pct_d)
        exits = (pct_k > 80) & (pct_k < pct_d)
    else:
        return None, None

    return entries, exits


def main() -> None:
    """Run the full feature selection pipeline."""
    config = load_instruments_config()
    pairs = config.get("instruments", {}).get("pairs", [])
    granularities = config.get("instruments", {}).get("granularities", ["H4"])
    # Use the primary analysis timeframe (2nd in list = H4)
    base_gran = granularities[1] if len(granularities) > 1 else "H4"

    for pair in pairs:
        run_sweep_for_pair(pair, base_gran)

    print("\n✅ Feature selection complete.")
    print("   ML pipeline will read config/features.toml automatically.\n")


if __name__ == "__main__":
    main()
