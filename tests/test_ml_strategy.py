import sys
import unittest
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

# Set path manually
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Mock Nautilus & joblib
from unittest.mock import patch

class TestMLSignalStrategy(unittest.TestCase):
    def setUp(self):
        # Patch sys.modules to mock nautilus_trader
        self.modules_patcher = patch.dict(sys.modules, {
            "nautilus_trader": MagicMock(),
            "nautilus_trader.config": MagicMock(),
            "nautilus_trader.model": MagicMock(),
            "nautilus_trader.model.data": MagicMock(),
            "nautilus_trader.model.enums": MagicMock(),
            "nautilus_trader.model.identifiers": MagicMock(),
            "nautilus_trader.model.objects": MagicMock(),
            "nautilus_trader.trading": MagicMock(),
            "nautilus_trader.trading.strategy": MagicMock(),
            "joblib": MagicMock(),
        })
        self.modules_patcher.start()

        # Setup Mock Strategy Base Class
        mock_strat_module = sys.modules["nautilus_trader.trading.strategy"]
        class MockStrategy:
            def __init__(self, config):
                self.config = config
                self.log = MagicMock()
                self.cache = MagicMock()
                self.order_factory = MagicMock()
            def subscribe_bars(self, bar_type):
                pass
        mock_strat_module.Strategy = MockStrategy

        # Setup Joblib
        mock_model = MagicMock()
        mock_model.predict.return_value = [1]
        sys.modules["joblib"].load.return_value = mock_model

        # Import Strategy (inside patch context)
        # We must reload it if it was already imported, or ensure fresh import
        if "titan.strategies.ml.strategy" in sys.modules:
            del sys.modules["titan.strategies.ml.strategy"]
        
        from titan.strategies.ml.strategy import MLSignalStrategy
        self.MLSignalStrategy = MLSignalStrategy

        # Create dummy parquet file
        self.data_dir = PROJECT_ROOT / "data"
        self.data_dir.mkdir(exist_ok=True)
        self.test_file = self.data_dir / "TEST_USD_H4.parquet"
        
        # ... rest of setup ...
        dates = pd.date_range(start="2023-01-01", periods=50, freq="4h")
        df = pd.DataFrame(
            {
                "open": [1.0] * 50,
                "high": [1.1] * 50,
                "low": [0.9] * 50,
                "close": [1.05] * 50,
                "volume": [100.0] * 50,
            },
            index=dates,
        )
        df.to_parquet(self.test_file)

        # Create Dummy Model File (to pass existence check)
        self.dummy_model_path = PROJECT_ROOT / "dummy.joblib"
        with open(self.dummy_model_path, "wb") as f:
            f.write(b"dummy")

        # Config
        self.mock_config = MagicMock()
        self.mock_config.model_path = str(self.dummy_model_path)
        self.mock_config.instrument_id = "TEST/USD"
        self.mock_config.bar_type = "TEST/USD-H4"
        self.mock_config.risk_pct = 0.02
        self.mock_config.warmup_bars = 50

    def tearDown(self):
        self.modules_patcher.stop()
        if self.test_file.exists():
            self.test_file.unlink()
        if self.dummy_model_path.exists():
            self.dummy_model_path.unlink()

    def test_warmup_and_signal(self):
        """Test full flow: warmup -> on_bar -> signal -> trade."""
        strategy = self.MLSignalStrategy(self.mock_config)

        # Manually invoke on_start (calls _warmup_history)
        # Note: on_start calls InstrumentId.from_str which returns a Mock.
        # But _warmup_history needs specific value.
        # We manually patch the instrument_id mock on the instance.
        strategy.instrument_id.value = "TEST/USD"

        strategy._warmup_history()

        # Should have loaded 50 bars
        self.assertEqual(len(strategy.history), 50)

        # Fill buffer to trigger prediction (need > 200)
        for i in range(200):
            strategy.history.append(
                {
                    "time": datetime.now(),
                    "open": 1.0,
                    "high": 1.1,
                    "low": 0.9,
                    "close": 1.0,
                    "volume": 100,
                }
            )

        # Mock order factory behavior
        strategy.order_factory.market.return_value = "order_obj"
        strategy.submit_order = MagicMock()
        strategy.close_all_positions = MagicMock()

        # Create Mock Bar
        mock_bar = MagicMock()
        mock_bar.close_time_as_datetime.return_value = datetime.now()
        mock_bar.open = Decimal("1.06")
        mock_bar.high = Decimal("1.07")
        mock_bar.low = Decimal("1.05")
        mock_bar.close = Decimal("1.065")
        mock_bar.volume = Decimal("110")

        # Run on_bar
        strategy.on_bar(mock_bar)

        # Check logs for "Analysis: Signal=..."
        logs = [str(call) for call in strategy.log.info.call_args_list]
        found_signal = any("Signal=1" in log for log in logs)
        self.assertTrue(found_signal, "Signal 1 not logged")

        # Check order submission
        self.assertTrue(strategy.submit_order.called, "Order not submitted")


if __name__ == "__main__":
    unittest.main()
