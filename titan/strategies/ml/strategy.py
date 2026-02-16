"""ml_strategy.py
-----------------

ML-driven strategy for NautilusTrader.
Loads a trained Joblib model, warms up with local data, calculates features
on streaming bars, and executes trades based on model predictions.
"""

from decimal import Decimal
from pathlib import Path

import joblib
import pandas as pd
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy

from titan.strategies.ml.features import build_features, load_feature_config


class MLSignalStrategyConfig(StrategyConfig):
    """Configuration for MLSignalStrategy."""

    model_path: str
    instrument_id: str
    bar_type: str  # e.g. "EUR/USD-H4"
    risk_pct: float = 0.02  # 2% risk per trade (placeholder)
    warmup_bars: int = 500  # Number of bars to load for history


class MLSignalStrategy(Strategy):
    """Executes trades based on a pre-trained ML model."""

    def __init__(self, config: MLSignalStrategyConfig):
        super().__init__(config)
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.bar_type = BarType.from_str(config.bar_type)

        # Load resources
        self.model = self._load_model(config.model_path)
        self.feature_config = load_feature_config(self.log)

        # History buffer (list of dicts, converted to DF for inference)
        self.history: list[dict] = []

    def _load_model(self, path: str):
        """Load the trained .joblib model."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Model not found at {p}")
        self.log.info(f"Loading model from {p}...")
        return joblib.load(p)

    def on_start(self):
        """Called when the strategy starts."""
        self.log.info("MLStrategy started.")

        # 1. Warmup History (Critical for immediate trading)
        self._warmup_history()

        # 2. Subscribe to Bars
        self.subscribe_bars(self.bar_type)
        self.log.info(f"Subscribed to bars: {self.bar_type}")

    def _warmup_history(self):
        """Load recent historical data from local Parquet to initialize indicators."""
        # Infer parquet filename from instrument and granularity
        # e.g. EUR/USD-H4 -> EUR_USD_H4.parquet
        pair = self.instrument_id.value.replace("/", "_")
        gran = self.bar_type.specifier.split("-")[-1] if "-" in self.bar_type.specifier else "H4"

        # Determine project root relative to this file
        # titan/strategies/ml/strategy.py -> .../Titan-Oanda
        project_root = Path(__file__).resolve().parents[3]
        parquet_path = project_root / "data" / f"{pair}_{gran}.parquet"

        if not parquet_path.exists():
            self.log.warning(f"⚠ Warmup file missing: {parquet_path}. Trading delayed.")
            return

        self.log.info(f"Loading warmup data from {parquet_path}...")
        try:
            df = pd.read_parquet(parquet_path)
            # Ensure index is datetime and sorted
            df = df.sort_index()
            # Take last N bars
            df = df.tail(self.config.warmup_bars)

            # Convert to list of dicts for our buffer
            # Expected cols: open, high, low, close, volume (lowercase)
            required_cols = ["open", "high", "low", "close", "volume"]
            if not all(c in df.columns for c in required_cols):
                self.log.error(f"Warmup data missing columns. Found: {df.columns}")
                return

            # Reset index to get 'time' column if it's the index
            df_reset = df.reset_index()
            # Rename index col to 'time' if needed, though we rely on order

            for _, row in df_reset.iterrows():
                # We store as floats in the buffer for feature calc
                self.history.append(
                    {
                        "time": row.iloc[0],  # Optimization: assume index is time
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row["volume"]),
                    }
                )

            self.log.info(f"Warmed up with {len(self.history)} bars.")

        except Exception as e:
            self.log.error(f"Failed to load warmup data: {e}")

    def on_bar(self, bar: Bar):
        """Called when a new bar closes."""
        # 1. Update History
        self.history.append(
            {
                "time": bar.close_time_as_datetime(),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
        )

        # Keep buffer size manageable (e.g. 1000 bars max, need enough for indicators)
        max_bars = self.config.warmup_bars + 200
        if len(self.history) > max_bars:
            self.history = self.history[-max_bars:]

        # 2. Check Warmup Status
        if len(self.history) < 200:  # Min required for slow SMAs (e.g. SMA 100)
            self.log.info(f"Waiting for history... ({len(self.history)}/200)")
            return

        # 3. Calculate Features
        try:
            df = pd.DataFrame(self.history)
            df.set_index("time", inplace=True)

            # Calculate features (identical to training)
            # TODO: Handle context_data (MTF) for live usage.
            # For this MVP, we pass empty context_data or handle it
            # if we have multi-bar subscriptions.
            # If the model relies on MTF, we must subscribe to those TFs too.
            # For now, we assume single-timeframe or missing MTF features won't crash (fill NaNs).

            context_data = {}  # Placeholder for MTF

            X = build_features(df, context_data, self.feature_config)

            # Get latest row
            latest_features = X.iloc[[-1]]  # Keep as DataFrame

            # 4. Predict
            signal = self.model.predict(latest_features)[0]

            # 5. Execute
            self._execute_signal(signal, bar.close)

        except Exception as e:
            self.log.error(f"Error in on_bar: {e}", exc_info=True)

    def _execute_signal(self, signal: int, price: Decimal):
        """Execute trades based on signal (1, -1, 0)."""
        # Get current position
        positions = self.cache.positions(instrument_id=self.instrument_id)
        position = positions[-1] if positions else None

        # Current Size (0 if no position)
        current_qty = position.quantity if position else Decimal(0)
        # Direction: 1 (Long), -1 (Short), 0 (Flat)
        current_dir = 0
        if position:
            current_dir = 1 if position.side == OrderSide.BUY else -1

        self.log.info(f"Analysis: Signal={signal}, CurrentPos={current_qty} ({current_dir})")

        # Logic:
        # Signal 1 (Long):
        #   - If Short -> Close Short, Open Long
        #   - If Flat -> Open Long
        #   - If Long -> Hold (or add, but simplistic: Hold)
        # Signal -1 (Short):
        #   - If Long -> Close Long, Open Short
        #   - If Flat -> Open Short
        #   - If Short -> Hold
        # Signal 0 (Flat):
        #   - Close any position

        qt = Decimal("1000")  # Fixed lot size for MVP

        if signal == 1:
            if current_dir == 1:
                return  # Already Long
            elif current_dir == -1:
                self.close_all_positions(self.instrument_id)
                self.buy(self.instrument_id, Quantity(qt, 0))
            else:  # Flat
                self.buy(self.instrument_id, Quantity(qt, 0))

        elif signal == -1:
            if current_dir == -1:
                return  # Already Short
            elif current_dir == 1:
                self.close_all_positions(self.instrument_id)
                self.sell(self.instrument_id, Quantity(qt, 0))
            else:  # Flat
                self.sell(self.instrument_id, Quantity(qt, 0))

        elif signal == 0:
            if current_dir != 0:
                self.close_all_positions(self.instrument_id)

    def buy(self, instrument_id, quantity):
        order = self.order_factory.market(
            instrument_id=instrument_id,
            order_side=OrderSide.BUY,
            quantity=quantity,
            time_in_force=TimeInForce.FOK,  # Fill or Kill for OANDA usually
        )
        self.submit_order(order)
        self.log.info(f"Submitted BUY {quantity} {instrument_id}")

    def sell(self, instrument_id, quantity):
        order = self.order_factory.market(
            instrument_id=instrument_id,
            order_side=OrderSide.SELL,
            quantity=quantity,
            time_in_force=TimeInForce.FOK,
        )
        self.submit_order(order)
        self.log.info(f"Submitted SELL {quantity} {instrument_id}")
