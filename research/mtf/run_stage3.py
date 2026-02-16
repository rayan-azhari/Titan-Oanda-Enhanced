"""run_mtf_stage3.py — Stage 3: Per-Timeframe MA & RSI Period Tuning.

Sweeps fast_ma, slow_ma, and rsi_period for each timeframe, starting
from the dominant D timeframe down to the least impactful W.

Fixed from Stage 1-2: SMA, threshold=0.10,
weights H1=0.10 H4=0.25 D=0.60 W=0.05.

Strategy: sweep each TF independently (greedy), in order of weight
importance: D -> H4 -> H1 -> W.  This avoids combinatorial explosion
while still tuning the most impactful parameters first.

Usage:
    uv run python research/mtf/run_stage3.py
"""

import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = PROJECT_ROOT / "config" / "mtf.toml"

try:
    import vectorbt as vbt
except ImportError:
    print("ERROR: vectorbt not installed.")
    sys.exit(1)

from titan.models.spread import build_spread_series

# ─────────────────────────────────────────────────────────────────────
# Fixed from earlier stages
# ─────────────────────────────────────────────────────────────────────
THRESHOLD = 0.10
WEIGHTS = {"H1": 0.10, "H4": 0.25, "D": 0.60, "W": 0.05}

# Sweep order: highest weight first
SWEEP_ORDER = ["D", "H4", "H1", "W"]

# Parameter grids per timeframe
PARAM_GRIDS = {
    "D": {
        "fast_ma": [5, 8, 10, 13, 15, 20],
        "slow_ma": [20, 25, 30, 40, 50, 60, 80],
        "rsi_period": [7, 10, 14, 21],
    },
    "H4": {
        "fast_ma": [10, 15, 20, 25, 30],
        "slow_ma": [30, 40, 50, 60, 80, 100],
        "rsi_period": [7, 10, 14, 21],
    },
    "H1": {
        "fast_ma": [10, 15, 20, 25, 30],
        "slow_ma": [30, 50, 80, 100, 150],
        "rsi_period": [7, 10, 14, 21, 28],
    },
    "W": {
        "fast_ma": [3, 5, 8, 10, 13],
        "slow_ma": [8, 13, 21, 26, 34],
        "rsi_period": [7, 10, 14, 21],
    },
}


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def load_data(pair: str, gran: str) -> pd.DataFrame | None:
    """Load parquet data."""
    path = DATA_DIR / f"{pair}_{gran}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    return df


def load_mtf_config() -> dict:
    """Load mtf.toml."""
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_tf_signal(close: pd.Series, fast_ma: int, slow_ma: int, rsi_period: int) -> pd.Series:
    """Directional signal for one TF using SMA."""
    fast = close.rolling(fast_ma).mean()
    slow = close.rolling(slow_ma).mean()
    rsi = compute_rsi(close, rsi_period)
    ma_sig = pd.Series(np.where(fast > slow, 0.5, -0.5), index=close.index)
    rsi_sig = pd.Series(np.where(rsi > 50, 0.5, -0.5), index=close.index)
    return ma_sig + rsi_sig


def extract_stats(pf) -> dict:
    """Pull metrics from VBT Portfolio."""
    n = pf.trades.count()
    return {
        "ret": pf.total_return(),
        "sharpe": pf.sharpe_ratio(),
        "dd": pf.max_drawdown(),
        "trades": n,
        "wr": float(pf.trades.win_rate()) if n > 0 else 0.0,
    }


def run_backtest(close, confluence, fees):
    """Run IS or OOS backtest, return long+short stats."""
    long = extract_stats(
        vbt.Portfolio.from_signals(
            close,
            entries=confluence >= THRESHOLD,
            exits=confluence < 0,
            init_cash=10_000,
            fees=fees,
            freq="4h",
        )
    )
    short = extract_stats(
        vbt.Portfolio.from_signals(
            close,
            entries=pd.Series(False, index=close.index),
            exits=pd.Series(False, index=close.index),
            short_entries=confluence <= -THRESHOLD,
            short_exits=confluence > 0,
            init_cash=10_000,
            fees=fees,
            freq="4h",
        )
    )
    return long, short


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def main() -> None:
    """Greedy per-TF parameter sweep."""
    pair = "EUR_USD"
    mtf_config = load_mtf_config()
    tfs = list(WEIGHTS.keys())

    print("=" * 60)
    print("  Stage 3: Per-Timeframe MA/RSI Period Tuning")
    print("=" * 60)
    print(f"  Sweep order: {' -> '.join(SWEEP_ORDER)}")
    print(f"  Weights: {WEIGHTS}")
    print(f"  Threshold: {THRESHOLD}")

    # Load primary + spread
    primary_df = load_data(pair, "H4")
    if primary_df is None:
        print("ERROR: H4 data missing.")
        sys.exit(1)
    primary_index = primary_df.index
    close = primary_df["close"]
    avg_spread = float(build_spread_series(primary_df, pair).mean())

    # IS/OOS split
    split = int(len(close) * 0.70)
    is_close, oos_close = close.iloc[:split], close.iloc[split:]

    # Load raw close per TF (for signal computation)
    tf_closes: dict[str, pd.Series] = {}
    for tf in tfs:
        df = load_data(pair, tf)
        if df is not None:
            tf_closes[tf] = df["close"]
            print(f"  {tf}: {len(df)} bars")

    # Start with current config as baseline
    best_params: dict[str, dict] = {}
    for tf in tfs:
        tf_cfg = mtf_config.get(tf, {})
        best_params[tf] = {
            "fast_ma": tf_cfg.get("fast_ma", 20),
            "slow_ma": tf_cfg.get("slow_ma", 50),
            "rsi_period": tf_cfg.get("rsi_period", 14),
        }

    all_results: list[dict] = []

    # Greedy sweep: tune one TF at a time, fix the others
    for sweep_tf in SWEEP_ORDER:
        if sweep_tf not in tf_closes:
            print(f"\n  Skipping {sweep_tf}: no data.")
            continue

        grid = PARAM_GRIDS[sweep_tf]
        combos = [
            {"fast_ma": f, "slow_ma": s, "rsi_period": r}
            for f in grid["fast_ma"]
            for s in grid["slow_ma"]
            for r in grid["rsi_period"]
            if f < s  # fast must be < slow
        ]

        print(f"\n{'=' * 50}")
        print(f"  Sweeping {sweep_tf} ({len(combos)} combos, w={WEIGHTS[sweep_tf]:.2f})")
        print(f"{'=' * 50}")

        # Pre-compute fixed TF signals (all TFs except the one being swept)
        fixed_signals: dict[str, pd.Series] = {}
        for tf in tfs:
            if tf == sweep_tf:
                continue
            if tf not in tf_closes:
                continue
            sig = compute_tf_signal(
                tf_closes[tf],
                best_params[tf]["fast_ma"],
                best_params[tf]["slow_ma"],
                best_params[tf]["rsi_period"],
            )
            fixed_signals[tf] = sig.reindex(primary_index, method="ffill") * WEIGHTS[tf]

        # Sum of fixed signals
        fixed_sum = sum(fixed_signals.values())

        best_cs = -999
        best_combo = None
        tf_results = []

        for i, combo in enumerate(combos):
            # Compute signal for the swept TF
            sig = compute_tf_signal(
                tf_closes[sweep_tf],
                combo["fast_ma"],
                combo["slow_ma"],
                combo["rsi_period"],
            )
            sig_resampled = sig.reindex(primary_index, method="ffill") * WEIGHTS[sweep_tf]

            confluence = fixed_sum + sig_resampled
            is_conf = confluence.iloc[:split]
            oos_conf = confluence.iloc[split:]

            is_long, is_short = run_backtest(is_close, is_conf, avg_spread)
            oos_long, oos_short = run_backtest(oos_close, oos_conf, avg_spread)

            lp = oos_long["sharpe"] / is_long["sharpe"] if is_long["sharpe"] != 0 else 0
            sp = oos_short["sharpe"] / is_short["sharpe"] if is_short["sharpe"] != 0 else 0
            cs = (
                is_long["sharpe"] + is_short["sharpe"] + oos_long["sharpe"] + oos_short["sharpe"]
            ) / 4

            row = {
                "sweep_tf": sweep_tf,
                "fast_ma": combo["fast_ma"],
                "slow_ma": combo["slow_ma"],
                "rsi_period": combo["rsi_period"],
                "is_long_ret": is_long["ret"],
                "is_long_sharpe": is_long["sharpe"],
                "is_short_ret": is_short["ret"],
                "is_short_sharpe": is_short["sharpe"],
                "oos_long_ret": oos_long["ret"],
                "oos_long_sharpe": oos_long["sharpe"],
                "oos_short_ret": oos_short["ret"],
                "oos_short_sharpe": oos_short["sharpe"],
                "long_parity": lp,
                "short_parity": sp,
                "combined_sharpe": cs,
            }
            tf_results.append(row)
            all_results.append(row)

            if lp > 0 and sp > 0 and cs > best_cs:
                best_cs = cs
                best_combo = combo

            if (i + 1) % 50 == 0:
                print(f"    [{i + 1}/{len(combos)}] done...")

        # Lock in best for this TF
        if best_combo:
            best_params[sweep_tf] = best_combo
            print(
                f"\n  Best {sweep_tf}: fast={best_combo['fast_ma']} "
                f"slow={best_combo['slow_ma']} rsi={best_combo['rsi_period']} "
                f"combined_sharpe={best_cs:.4f}"
            )
        else:
            print(f"  No improvement for {sweep_tf}, keeping defaults.")

    # ── Save full results ──
    df = pd.DataFrame(all_results)
    csv_path = REPORTS_DIR / "mtf_stage3_scoreboard.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nFull scoreboard: {csv_path}")

    # ── Final report ──
    print("\n" + "=" * 60)
    print("  FINAL OPTIMISED PARAMETERS")
    print("=" * 60)
    for tf in SWEEP_ORDER:
        p = best_params[tf]
        print(
            f"  {tf:3s}: fast_ma={p['fast_ma']:3d}  "
            f"slow_ma={p['slow_ma']:3d}  "
            f"rsi={p['rsi_period']:2d}"
        )

    # Run final combined backtest with all best params
    print("\n--- Final Backtest with All Optimised Params ---")
    signals_sum = pd.Series(0.0, index=primary_index)
    for tf in tfs:
        if tf not in tf_closes:
            continue
        sig = compute_tf_signal(
            tf_closes[tf],
            best_params[tf]["fast_ma"],
            best_params[tf]["slow_ma"],
            best_params[tf]["rsi_period"],
        )
        signals_sum += sig.reindex(primary_index, method="ffill") * WEIGHTS[tf]

    is_conf = signals_sum.iloc[:split]
    oos_conf = signals_sum.iloc[split:]

    is_long, is_short = run_backtest(is_close, is_conf, avg_spread)
    oos_long, oos_short = run_backtest(oos_close, oos_conf, avg_spread)

    cs = (is_long["sharpe"] + is_short["sharpe"] + oos_long["sharpe"] + oos_short["sharpe"]) / 4
    lp = oos_long["sharpe"] / is_long["sharpe"] if is_long["sharpe"] != 0 else 0
    sp = oos_short["sharpe"] / is_short["sharpe"] if is_short["sharpe"] != 0 else 0

    print(f"  Combined Sharpe: {cs:.4f}")
    print(
        f"  IS  LONG:   ret={is_long['ret']:.2%}  "
        f"sharpe={is_long['sharpe']:.3f}  "
        f"dd={is_long['dd']:.2%}  "
        f"trades={is_long['trades']}"
    )
    print(
        f"  IS  SHORT:  ret={is_short['ret']:.2%}  "
        f"sharpe={is_short['sharpe']:.3f}  "
        f"dd={is_short['dd']:.2%}  "
        f"trades={is_short['trades']}"
    )
    print(
        f"  OOS LONG:   ret={oos_long['ret']:.2%}  "
        f"sharpe={oos_long['sharpe']:.3f}  "
        f"dd={oos_long['dd']:.2%}  "
        f"trades={oos_long['trades']}"
    )
    print(
        f"  OOS SHORT:  ret={oos_short['ret']:.2%}  "
        f"sharpe={oos_short['sharpe']:.3f}  "
        f"dd={oos_short['dd']:.2%}  "
        f"trades={oos_short['trades']}"
    )
    print(f"  Parity: L={lp:.2f}  S={sp:.2f}")

    # ── Save optimised config ──
    config_lines = [
        "# " + "=" * 56,
        "# mtf.toml -- Multi-Timeframe Confluence Configuration",
        "# " + "=" * 56,
        "# Optimised via Stage 1-3 sweeps (run_mtf_optimisation.py,",
        "# run_mtf_stage2.py, run_mtf_stage3.py).",
        "",
        "[weights]",
        f"H1 = {WEIGHTS['H1']:.2f}",
        f"H4 = {WEIGHTS['H4']:.2f}",
        f"D  = {WEIGHTS['D']:.2f}",
        f"W  = {WEIGHTS['W']:.2f}",
        "",
        f"confirmation_threshold = {THRESHOLD:.2f}",
        "",
    ]
    for tf in ["H1", "H4", "D", "W"]:
        p = best_params[tf]
        config_lines += [
            f"[{tf}]",
            f"fast_ma = {p['fast_ma']}",
            f"slow_ma = {p['slow_ma']}",
            f"rsi_period = {p['rsi_period']}",
            "",
        ]

    config_text = "\n".join(config_lines)
    CONFIG_PATH.write_text(config_text, encoding="utf-8")
    print(f"\nSaved optimised config to {CONFIG_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()
