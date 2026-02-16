"""run_backtesting_validation.py — Secondary validation using Backtesting.py.

Provides bar-by-bar processing validation of strategies, generating
HTML trade plots for visual audit against VBT results.

Directive: Strategy Validation (Backtesting.py).md
"""

import sys
import tomllib
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DATA_DIR = PROJECT_ROOT / ".tmp" / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    from backtesting import Backtest, Strategy  # noqa: F401, E402
except ImportError:
    print("ERROR: backtesting is not installed. Run `uv sync` first.")
    sys.exit(1)


def load_strategy_config() -> dict:
    """Load strategy parameters from config/strategy_config.toml."""
    config_path = PROJECT_ROOT / "config" / "strategy_config.toml"
    if not config_path.exists():
        print(f"ERROR: {config_path} not found.")
        sys.exit(1)
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def load_parquet_for_backtesting(pair: str, granularity: str) -> pd.DataFrame:
    """Load raw Parquet and convert to the Backtesting.py DataFrame format.

    Backtesting.py requires columns: Open, High, Low, Close, Volume
    with a DatetimeIndex.

    Args:
        pair: Instrument name (e.g., "EUR_USD").
        granularity: Candle granularity (e.g., "M5").

    Returns:
        DataFrame in Backtesting.py format.
    """
    path = RAW_DATA_DIR / f"{pair}_{granularity}.parquet"
    if not path.exists():
        print(f"ERROR: {path} not found. Run download_oanda_data.py first.")
        sys.exit(1)

    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    # Convert to Backtesting.py naming convention
    df = df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )

    # Convert Decimal to float
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col].astype(float)

    return df


class RSIStrategy(Strategy):
    """RSI-based mean-reversion strategy for validation.

    Buys when RSI crosses below the entry threshold (oversold),
    sells when RSI crosses above the exit threshold (overbought).
    """

    rsi_window = 14
    rsi_entry = 30
    rsi_exit = 70

    def init(self):
        """Initialise RSI indicator."""
        close = pd.Series(self.data.Close)
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(self.rsi_window).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(self.rsi_window).mean()
        rs = gain / loss
        self.rsi = self.I(lambda: 100 - (100 / (1 + rs)), name="RSI")

    def next(self):
        """Execute trading logic on each bar."""
        if self.rsi[-1] < self.rsi_entry and not self.position:
            self.buy()
        elif self.rsi[-1] > self.rsi_exit and self.position:
            self.position.close()


def main() -> None:
    """Run Backtesting.py validation and generate HTML report."""
    config = load_strategy_config()

    rsi_cfg = config.get("rsi", {})
    pair = config.get("instrument", "EUR_USD")
    granularity = config.get("granularity", "M5")

    print(f"📋 Validating strategy on {pair} ({granularity}) via Backtesting.py\n")

    df = load_parquet_for_backtesting(pair, granularity)

    # Configure strategy from TOML
    RSIStrategy.rsi_window = rsi_cfg.get("window", 14)
    RSIStrategy.rsi_entry = rsi_cfg.get("entry", 30)
    RSIStrategy.rsi_exit = rsi_cfg.get("exit", 70)

    bt = Backtest(
        df,
        RSIStrategy,
        cash=10_000,
        commission=0.0002,
        exclusive_orders=True,
    )

    stats = bt.run()
    print(stats)

    # Generate HTML plot for visual audit
    plot_path = REPORTS_DIR / f"backtesting_validation_{pair}.html"
    bt.plot(filename=str(plot_path), open_browser=False)
    print(f"\n  📊 HTML report saved to {plot_path}")
    print("  → Manually inspect trade entries for logic alignment with VBT.")

    print("\n✅ Backtesting.py validation complete.\n")


if __name__ == "__main__":
    main()
