"""run_vbt_optimisation.py — Vectorised strategy optimisation using VectorBT (free).

Runs parameterised strategy backtests across multiple indicator ranges,
generates Sharpe Ratio heatmaps via plotly, and exports optimal parameters
to config/strategy_config.toml.

Includes in-sample / out-of-sample split to detect overfitting.

Directive: Alpha Research Loop (VectorBT).md
"""

import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    import vectorbt as vbt
except ImportError:
    print("ERROR: vectorbt is not installed. Run `uv sync` first.")
    sys.exit(1)

try:
    import plotly.graph_objects as go
except ImportError:
    print("ERROR: plotly is not installed. Run `uv sync` first.")
    sys.exit(1)

from titan.models.spread import build_spread_series

# Mapping OANDA granularity codes to pandas frequency strings.
# VBT needs this to calculate annualised Sharpe ratios.
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


def load_instruments_config() -> dict:
    """Load the instruments configuration from config/instruments.toml."""
    config_path = PROJECT_ROOT / "config" / "instruments.toml"
    if not config_path.exists():
        print(f"ERROR: {config_path} not found.")
        sys.exit(1)
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def load_raw_data(pair: str, granularity: str) -> pd.DataFrame | None:
    """Load raw Parquet data for a given instrument and granularity.

    Args:
        pair: Instrument name (e.g., "EUR_USD").
        granularity: Candle granularity (e.g., "H4", "D").

    Returns:
        DataFrame with timestamp-indexed OHLCV data, or None if not found.
    """
    path = RAW_DATA_DIR / f"{pair}_{granularity}.parquet"
    if not path.exists():
        print(f"  ⚠ {path.name} not found — skipping. Run fetch script first.")
        return None
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    # Convert Decimal columns to float for VBT compatibility
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    # Set index frequency for VBT annualisation
    freq = GRAN_TO_FREQ.get(granularity)
    if freq:
        df = df.asfreq(freq, method="pad")
    return df


def split_in_out_of_sample(
    df: pd.DataFrame, is_ratio: float = 0.70
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into in-sample and out-of-sample.

    Args:
        df: Full OHLCV DataFrame.
        is_ratio: Fraction of data for in-sample (default 70%).

    Returns:
        Tuple of (in_sample_df, out_of_sample_df).
    """
    split_idx = int(len(df) * is_ratio)
    is_df = df.iloc[:split_idx]
    oos_df = df.iloc[split_idx:]
    print(f"    IS: {len(is_df)} bars ({is_df.index.min()} → {is_df.index.max()})")
    print(f"    OOS: {len(oos_df)} bars ({oos_df.index.min()} → {oos_df.index.max()})")
    return is_df, oos_df


def run_rsi_optimisation(
    close: pd.Series,
    rsi_windows: list[int],
    entry_thresholds: list[int],
    fees: float = 0.0003,
):
    """Run parameterised RSI strategy optimisation.

    Uses vectorbt's built-in RSI indicator and Portfolio.from_signals
    to sweep across RSI window and entry threshold combinations.

    Args:
        close: Close price series.
        rsi_windows: List of RSI lookback periods to test.
        entry_thresholds: List of RSI entry thresholds to test.
        fees: Transaction cost as a decimal (e.g., 0.0003 = 3 pips).

    Returns:
        VBT Portfolio object with all parameter combinations.
    """
    rsi = vbt.RSI.run(close, window=rsi_windows, param_product=True)

    entries = rsi.rsi_crossed_below(entry_thresholds)
    exits = rsi.rsi_crossed_above(100 - np.array(entry_thresholds))

    portfolio = vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=exits,
        init_cash=10_000,
        fees=fees,
    )
    return portfolio


def generate_sharpe_heatmap(sharpe_df: pd.DataFrame, pair: str, label: str = "IS") -> Path:
    """Generate and save a Sharpe Ratio heatmap using plotly.

    Args:
        sharpe_df: 2D DataFrame of Sharpe values (rows=windows, cols=thresholds).
        pair: Instrument name for titling.
        label: "IS" or "OOS" label.

    Returns:
        Path to the saved heatmap HTML file.
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=sharpe_df.values,
            x=[str(c) for c in sharpe_df.columns],
            y=[str(r) for r in sharpe_df.index],
            colorscale="RdYlGn",
            colorbar=dict(title="Sharpe"),
        )
    )
    fig.update_layout(
        title=f"Sharpe Ratio Heatmap — {pair} ({label})",
        xaxis_title="RSI Entry Threshold",
        yaxis_title="RSI Window",
    )

    heatmap_path = REPORTS_DIR / f"sharpe_heatmap_{pair}_{label.lower()}.html"
    fig.write_html(str(heatmap_path))
    print(f"  📊 Heatmap saved to {heatmap_path}")
    return heatmap_path


def sharpe_to_2d(sharpe_series, rsi_windows, entry_thresholds) -> pd.DataFrame:
    """Reshape a flat Sharpe series into a 2D DataFrame for heatmap.

    Args:
        sharpe_series: Flat Sharpe series from VBT portfolio.
        rsi_windows: RSI window values (row labels).
        entry_thresholds: Entry threshold values (column labels).

    Returns:
        2D DataFrame with windows as rows, thresholds as columns.
    """
    values = sharpe_series.values.reshape(len(rsi_windows), len(entry_thresholds))
    return pd.DataFrame(values, index=rsi_windows, columns=entry_thresholds)


def find_plateau_candidates(sharpe_2d: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Find parameters in the 'Plateau of Stability'.

    Selects candidates where neighbours are also profitable,
    indicating robustness rather than curve-fitting.

    Args:
        sharpe_2d: 2D Sharpe DataFrame.
        top_n: Number of top candidates to return.

    Returns:
        DataFrame of top candidates with their neighbour averages.
    """
    scores = []
    for i in range(1, sharpe_2d.shape[0] - 1):
        for j in range(1, sharpe_2d.shape[1] - 1):
            centre = sharpe_2d.iloc[i, j]
            neighbours = sharpe_2d.iloc[i - 1 : i + 2, j - 1 : j + 2].values.flatten()
            neighbour_avg = np.mean(neighbours)
            neighbour_min = np.min(neighbours)
            scores.append(
                {
                    "rsi_window": sharpe_2d.index[i],
                    "entry_threshold": sharpe_2d.columns[j],
                    "sharpe": centre,
                    "neighbour_avg": neighbour_avg,
                    "neighbour_min": neighbour_min,
                    "stability_score": neighbour_min,  # Worst neighbour = robustness
                }
            )

    results = pd.DataFrame(scores)
    if results.empty:
        return results
    return results.sort_values("stability_score", ascending=False).head(top_n)


def main() -> None:
    """Run the full VectorBT optimisation loop with OOS validation."""
    config = load_instruments_config()
    pairs = config.get("instruments", {}).get("pairs", [])
    granularity = config.get("instruments", {}).get("granularities", ["H4"])[0]

    # Parameter ranges for RSI strategy (swing trading)
    rsi_windows = list(range(10, 30))  # RSI 10–29
    entry_thresholds = list(range(20, 40))  # Entry at RSI 20–39

    for pair in pairs:
        print(f"\n{'=' * 60}")
        print(f"🔬 Optimising {pair} ({granularity})")
        print(f"{'=' * 60}\n")

        df = load_raw_data(pair, granularity)
        if df is None:
            continue
        is_df, oos_df = split_in_out_of_sample(df)

        # Calculate session-weighted average spread for this pair
        spread_series = build_spread_series(df, pair)
        avg_spread = float(spread_series.mean())
        print(f"  📊 Using session-weighted spread: {avg_spread * 10000:.1f} pips\n")

        # --- In-Sample Optimisation ---
        print("  ▶ In-Sample Optimisation...")
        is_portfolio = run_rsi_optimisation(
            is_df["close"], rsi_windows, entry_thresholds, fees=avg_spread
        )
        is_sharpe = is_portfolio.sharpe_ratio()
        is_sharpe_2d = sharpe_to_2d(is_sharpe, rsi_windows, entry_thresholds)
        generate_sharpe_heatmap(is_sharpe_2d, pair, "IS")

        # --- Plateau candidates ---
        candidates = find_plateau_candidates(is_sharpe_2d)
        if candidates.empty:
            print(f"  ⚠️  No stable candidates for {pair}. Skipping OOS.")
            continue

        print("\n  🏆 Top IS Plateau Candidates:")
        for _, row in candidates.iterrows():
            print(
                f"     RSI({int(row['rsi_window'])}) < {int(row['entry_threshold'])}"
                f"  Sharpe={row['sharpe']:.3f}  NeighMin={row['neighbour_min']:.3f}"
            )

        # --- Out-of-Sample Validation ---
        print("\n  ▶ Out-of-Sample Validation...")
        oos_portfolio = run_rsi_optimisation(
            oos_df["close"], rsi_windows, entry_thresholds, fees=avg_spread
        )
        oos_sharpe = oos_portfolio.sharpe_ratio()
        oos_sharpe_2d = sharpe_to_2d(oos_sharpe, rsi_windows, entry_thresholds)
        generate_sharpe_heatmap(oos_sharpe_2d, pair, "OOS")

        # --- Parity Check ---
        print("\n  ▶ IS vs OOS Parity Check:")
        for _, row in candidates.iterrows():
            w = int(row["rsi_window"])
            t = int(row["entry_threshold"])
            is_val = is_sharpe_2d.loc[w, t]
            oos_val = oos_sharpe_2d.loc[w, t]
            ratio = oos_val / is_val if is_val != 0 else 0
            status = "✓ PASS" if ratio >= 0.5 else "✗ OVERFIT"
            print(
                f"     RSI({w})<{t}: IS={is_val:.3f} OOS={oos_val:.3f} Ratio={ratio:.2f} {status}"
            )

    print("\n✅ VBT optimisation complete.")
    print("   Transfer validated results to config/strategy_config.toml.\n")


if __name__ == "__main__":
    main()
