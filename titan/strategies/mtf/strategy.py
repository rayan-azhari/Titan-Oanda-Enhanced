"""mtf_strategy.py
-----------------

Multi-Timeframe Confluence Strategy for NautilusTrader.
Implements the "Signal Only" logic with Volatility-Adjusted Risk Sizing.
"""

import tomllib
from decimal import Decimal
from pathlib import Path

import pandas as pd
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.datetime import unix_nanos_to_dt
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy

# Reuse robust indicator implementations
from titan.strategies.ml.features import atr, ema, rsi, sma, wma


class MTFConfluenceConfig(StrategyConfig):
    """Configuration for MTF Strategy."""

    instrument_id: str
    bar_types: dict[str, str]  # Map: "H1": "EUR/USD-1h", etc.
    config_path: str = "config/mtf.toml"
    risk_pct: float = 0.01  # 1% Risk per trade
    leverage_cap: float = 5.0  # Max 5x leverage
    warmup_bars: int = 1000  # History to load


class MTFConfluenceStrategy(Strategy):
    """Executes trades based on Multi-Timeframe Confluence.

    Logic:
    1.  Subscribe to H1, H4, D, W bars.
    2.  On every bar close, update history and re-calculate Signal for that TF.
    3.  Compute Weighted Confluence Score.
    4.  Signal-Based Exits (No Trailing Stop).
    5.  Sizing: 1% Equity Risk per 2 ATR move.
    """

    def __init__(self, config: MTFConfluenceConfig):
        super().__init__(config)

        # Load optimization params
        self.toml_cfg = self._load_toml(config.config_path)

        # Identifier
        self.instrument_id = InstrumentId.from_str(config.instrument_id)

        # Map BarType -> TF Key (e.g. "EUR/USD-1h" -> "H1")
        self.bar_type_map = {}
        for tf, periodicity in config.bar_types.items():
            bt = BarType.from_str(periodicity)
            self.bar_type_map[bt] = tf

        # State buffers
        self.history = {tf: [] for tf in ["M5", "H1", "H4", "D", "W"]}

        # Current Signal State (-1.0 to +1.0) for each TF
        # Initialize to 0 (Neutral)
        self.signals = {tf: 0.0 for tf in ["M5", "H1", "H4", "D", "W"]}

        # Raw indicator values for the status dashboard
        self.indicator_state = {
            tf: {"fast_ma": None, "slow_ma": None, "rsi": None}
            for tf in ["M5", "H1", "H4", "D", "W"]
        }

        # Latest ATR (H1 or used TF) for volatility-adjusted sizing
        self.latest_atr = None

    def _load_toml(self, path: str) -> dict:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config not found at {p}")
        with open(p, "rb") as fobj:
            return tomllib.load(fobj)

    def on_start(self):
        """Lifecycle: Strategy Started."""
        self.log.info("MTF Strategy Started. Warming up...")

        # Subscribe
        for bt in self.bar_type_map.keys():
            self.subscribe_bars(bt)

        # Warmup
        self._warmup_all()

        self.log.info("Warmup complete. Ready for signals.")

    def _warmup_all(self):
        """Load history for all timeframes."""
        # titan/strategies/mtf/strategy.py -> mtf -> strategies -> titan -> ROOT
        project_root = Path(__file__).resolve().parents[3]
        data_dir = project_root / "data"

        self.log.info(f"Looking for warmup data in: {data_dir}")

        # Iterate over our TFs
        for bt, tf in self.bar_type_map.items():
            # Infer filename: EUR_USD_H1.parquet
            # Parse spec from BarType object
            spec = bt.spec
            agg = spec.aggregation
            interval = spec.step
            
            # Robust extraction of aggregation string
            agg_str = str(agg)
            # Handle enum objects like Aggregation.MINUTE
            if hasattr(agg, "name"):
                agg_str = agg.name
            elif hasattr(agg, "value"):
                agg_str = str(agg.value)
            
            agg_str = agg_str.upper()

            suffix = "UNKNOWN"
            # Logic: If it's a 5-minute bar, it maps to M5.
            # Check interval first if Agg is MINUTE
            if "MINUTE" in agg_str:
                if interval == 5:
                    suffix = "M5"
                # Fallback: sometimes Aggregation is just MINUTE and relies on step
            elif "HOUR" in agg_str:
                if interval == 1:
                    suffix = "H1"
                elif interval == 4:
                    suffix = "H4"
            elif "DAY" in agg_str:
                suffix = "D"
            elif "WEEK" in agg_str:
                suffix = "W"
            
            # Additional Fallback based on BarType string (e.g. "EUR/USD-5-MINUTE-MID-INTERNAL")
            if suffix == "UNKNOWN":
                 bt_str = str(bt).upper()
                 if "5-MINUTE" in bt_str:
                     suffix = "M5"
                 elif "1-HOUR" in bt_str:
                     suffix = "H1"
                 elif "4-HOUR" in bt_str:
                     suffix = "H4"
                 elif "1-DAY" in bt_str:
                     suffix = "D"

            if suffix == "UNKNOWN":
                msg = f"Unknown BarType spec usage: {bt} (Agg: {agg_str}, Interval: {interval})"
                self.log.warning(msg)
                continue

            pair_str = self.instrument_id.symbol.value.replace("/", "_")
            parquet_path = data_dir / f"{pair_str}_{suffix}.parquet"

            if not parquet_path.exists():
                msg = f"Missing warmup file: {parquet_path}"
                self.log.warning(msg)
                continue

            self.log.info(f"Loading {tf} warmup from {parquet_path}")
            try:
                df = pd.read_parquet(parquet_path).sort_index().tail(self.config.warmup_bars)
            except Exception as e:
                self.log.error(f"Failed to load parquet: {e}")
                continue

            # Populate history
            for t, row in df.iterrows():
                self.history[tf].append(
                    {
                        "time": t,
                        "close": float(row["close"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                    }
                )

            # Initial Signal Calc
            self.log.info(f"DEBUG: Calculating signal for {tf}...")
            self._update_signal(tf)

        self.log.info("DEBUG: Warmup loop finished.")

    def on_bar(self, bar: Bar):
        """Lifecycle: New Bar Closed."""
        tf = self.bar_type_map.get(bar.bar_type)
        if not tf:
            return  # Unknown bar type

        # Update History
        self.history[tf].append(
            {
                "time": unix_nanos_to_dt(bar.ts_event),
                "close": float(bar.close),
                "high": float(bar.high),
                "low": float(bar.low),
            }
        )

        # Limit buffer
        if len(self.history[tf]) > self.config.warmup_bars + 100:
            self.history[tf] = self.history[tf][-self.config.warmup_bars :]

        # Recalculate Logic
        self._update_signal(tf)

        # Evaluate Confluence (Trigger execution only on fastest TF usually?
        # Or any? If D flips, we want to know. So Any.)
        self._evaluate_confluence(price=bar.close)

    def _update_signal(self, tf: str):
        """Calculate Technical Signal for a specific Timeframe."""
        data = self.history[tf]
        if len(data) < 50:  # Min needed for slow SMA
            return

        df = pd.DataFrame(data)
        close = df["close"]

        # Load params
        params = self.toml_cfg.get(tf, {})
        fast_p = params.get("fast_ma", 10)
        slow_p = params.get("slow_ma", 20)
        rsi_p = params.get("rsi_period", 14)
        
        # Determine MA Type (default SMA for backward compat)
        ma_type = self.toml_cfg.get("ma_type", "SMA").upper()

        # Calc Indicators
        if ma_type == "EMA":
            s_fast = ema(close, fast_p)
            s_slow = ema(close, slow_p)
        elif ma_type == "WMA":
            s_fast = wma(close, fast_p)
            s_slow = wma(close, slow_p)
        else: # SMA
            s_fast = sma(close, fast_p)
            s_slow = sma(close, slow_p)
            
        r = rsi(close, rsi_p)

        # Store ATR if this is H1 (for sizing)
        if tf == "H1":
            # Recalc ATR
            # We need high/low in DF
            df["high"] = pd.Series([d["high"] for d in data], index=df.index)
            df["low"] = pd.Series([d["low"] for d in data], index=df.index)
            # Use standard 14 ATR for sizing volatility measure
            atr_s = atr(df, 14)
            if not atr_s.empty:
                self.latest_atr = float(atr_s.iloc[-1])

        # Calc Signal Components (Last bar)
        last_fast = s_fast.iloc[-1]
        last_slow = s_slow.iloc[-1]
        last_rsi = r.iloc[-1]

        trend_score = 0.5 if last_fast > last_slow else -0.5
        mom_score = 0.5 if last_rsi > 50 else -0.5

        # Store for dashboard
        self.indicator_state[tf] = {
            "fast_ma": last_fast,
            "slow_ma": last_slow,
            "rsi": last_rsi,
        }

        # Total Signal
        self.signals[tf] = trend_score + mom_score  # Range: -1.0 to +1.0

    def _evaluate_confluence(self, price: Decimal):
        """Combine signals and execute."""
        weights = self.toml_cfg.get("weights", {"H1": 0.1, "H4": 0.25, "D": 0.6, "W": 0.05})
        threshold = self.toml_cfg.get("confirmation_threshold", 0.10)

        score = 0.0
        for tf, weight in weights.items():
            score += self.signals[tf] * weight

        # Determine Bias
        if score >= threshold:
            bias = 1
            signal_label = "LONG"
        elif score <= -threshold:
            bias = -1
            signal_label = "SHORT"
        else:
            bias = 0
            signal_label = "FLAT"

        # Current position state
        positions = self.cache.positions(instrument_id=self.instrument_id)
        position = positions[-1] if positions else None
        pos_label = "FLAT"
        if position and position.is_open:
            pos_label = "LONG" if str(position.side) == "LONG" else "SHORT"

        # --- Status Dashboard ---
        self._log_status_dashboard(
            price=price,
            score=score,
            threshold=threshold,
            signal_label=signal_label,
            pos_label=pos_label,
            weights=weights,
        )

        # Execute
        self._execute_bias(bias, price)

    def _log_status_dashboard(
        self,
        price: Decimal,
        score: float,
        threshold: float,
        signal_label: str,
        pos_label: str,
        weights: dict,
    ):
        """Print a formatted multi-timeframe status dashboard."""
        sep = "═" * 55
        lines = [f"\n{sep}"]
        lines.append(f"  MTF STATUS @ Price: {float(price):.5f}")
        lines.append(f"{'─' * 55}")

        if not self.signals:
            return

        # Sort TFs for display (Slowest to Fastest usually looks best, or config order)
        # We can use the keys from self.signals, filtered by what's actually in use
        # or just hardcode the expected set for 5m strategy: D, H4, H1, M5
        display_tfs = [tf for tf in ["D", "H4", "H1", "M5", "W"] if tf in self.signals]

        for tf in display_tfs:
            st = self.indicator_state.get(tf)
            if not st:
                continue

            sig = self.signals[tf]
            w = weights.get(tf, 0)

            if st["fast_ma"] is not None:
                sma_dir = "BULL" if st["fast_ma"] > st["slow_ma"] else "BEAR"
                rsi_val = f"{st['rsi']:.1f}"
            else:
                sma_dir = " ?? "
                rsi_val = " ? "

            weighted = sig * w
            lines.append(
                f"  {tf:>2}  │  MA:  {sma_dir}  │  RSI: {rsi_val:>5}"
                f"  │  Signal: {sig:+.1f}  │  Weighted: {weighted:+.3f}"
            )

        lines.append(f"{'─' * 55}")
        lines.append(
            f"  CONFLUENCE: {score:+.3f}  │  Threshold: ±{threshold}  │  Signal: {signal_label}"
        )
        lines.append(
            f"  Position: {pos_label}  │  ATR(14): {self.latest_atr:.5f}"
            if self.latest_atr
            else f"  Position: {pos_label}  │  ATR: pending"
        )
        lines.append(sep)

        self.log.info("\n".join(lines))

    def _execute_bias(self, bias: int, price: Decimal):
        """Manage Positions based on Bias (Signal Only)."""
        instrument_id = self.instrument_id
        positions = self.cache.positions(instrument_id=instrument_id)
        position = positions[-1] if positions else None

        # Current State
        current_dir = 0
        if position:
            current_dir = 1 if position.side == OrderSide.BUY else -1

        # 1. Check Entry/Reversal
        if bias == 1:  # WANT LONG
            if current_dir == 1:
                return  # Hold
            elif current_dir == -1:
                self.log.info("Signal Flip: Short -> Long. Closing Short.")
                self.close_all_positions(instrument_id)
                self._open_position(OrderSide.BUY, price)
            else:  # Flat
                self.log.info("Signal Entry: Long.")
                self._open_position(OrderSide.BUY, price)

        elif bias == -1:  # WANT SHORT
            if current_dir == -1:
                return  # Hold
            elif current_dir == 1:
                self.log.info("Signal Flip: Long -> Short. Closing Long.")
                self.close_all_positions(instrument_id)
                self._open_position(OrderSide.SELL, price)
            else:  # Flat
                self.log.info("Signal Entry: Short.")
                self._open_position(OrderSide.SELL, price)

        elif bias == 0:  # NEUTRAL
            if current_dir != 0:
                self.log.info(f"Signal Neutral. Closing position ({current_dir}).")
                self.close_all_positions(instrument_id)

    def _open_position(self, side: OrderSide, price: Decimal):
        """Calculate Size and Submit Order."""
        if self.latest_atr is None or self.latest_atr == 0:
            self.log.warning("ATR not yet ready. Skipping trade.")
            return

        # 1% Risk Sizing
        # Units = (Equity * Risk%) / (2 * ATR)

        # Get Account (Dynamic lookup)
        # In live mode, we should look for the account associated with the execution client or just grab the first one.
        # self.cache.accounts() returns a list of Account objects directly.
        accounts = self.cache.accounts()
        if not accounts:
            self.log.error("No account found in cache! Cannot size position.")
            return

        account = accounts[0]
        
        # Use balance_total() to get Money object, then cast to float
        # This includes free + locked (Equity)
        equity_money = account.balance_total()
        equity = float(equity_money.as_double())

        risk_amount = equity * self.config.risk_pct
        stop_loss_dist = float(self.latest_atr) * 2.0

        if stop_loss_dist == 0:
            return

        raw_units = risk_amount / stop_loss_dist

        # Cap Leverage
        if float(price) > 0:
            max_units = (equity * self.config.leverage_cap) / float(price)
            units = min(raw_units, max_units)
        else:
            units = 0

        # Round to integer (Oanda requirement usually, or lots?)
        # Oanda accepts units (int).
        qty = Quantity.from_int(int(units))

        if int(units) <= 0:
            self.log.warning("Calculated size is 0.")
            return

        # Place Order
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=side,
            quantity=qty,
            time_in_force=TimeInForce.FOK,
        )
        self.submit_order(order)
        self.log.info(f"Submitted {side} {qty} (ATR: {self.latest_atr:.5f})")
