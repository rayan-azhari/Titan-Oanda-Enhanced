
import sys
import unittest
from unittest.mock import MagicMock, patch
from decimal import Decimal
from pathlib import Path

# Set path manually
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

class TestMTFStrategyFix(unittest.TestCase):
    def setUp(self):
        # Patch sys.modules to mock nautilus_trader
        self.modules_patcher = patch.dict(
            sys.modules,
            {
                "nautilus_trader": MagicMock(),
                "nautilus_trader.core": MagicMock(),
                "nautilus_trader.core.datetime": MagicMock(),
                "nautilus_trader.config": MagicMock(),
                "nautilus_trader.model": MagicMock(),
                "nautilus_trader.model.data": MagicMock(),
                "nautilus_trader.model.enums": MagicMock(),
                "nautilus_trader.model.identifiers": MagicMock(),
                "nautilus_trader.model.objects": MagicMock(),
                "nautilus_trader.trading": MagicMock(),
                "nautilus_trader.trading.strategy": MagicMock(),
                "tomllib": MagicMock(), # Mock tomllib to avoid file reading issues
            },
        )
        self.modules_patcher.start()

        # Mock Strategy Base Class
        mock_strat_module = sys.modules["nautilus_trader.trading.strategy"]
        
        class MockStrategy:
            def __init__(self, config):
                self.config = config
                self.log = MagicMock()
                self.cache = MagicMock()
                self.order_factory = MagicMock()
                self.instrument_id = MagicMock() # Mock instrument_id
                self.instrument_id.value = "EUR/USD"

            def subscribe_bars(self, bar_type):
                pass

            def submit_order(self, order):
                pass

            def close_all_positions(self, instrument_id):
                pass

        mock_strat_module.Strategy = MockStrategy

        # Mock tomllib load
        sys.modules["tomllib"].load.return_value = {
            "weights": {"H1": 0.1, "H4": 0.25, "D": 0.6, "W": 0.05},
            "confirmation_threshold": 0.10
        }

        # Import Strategy (inside patch context)
        # Reload to ensure mocks are used
        if "titan.strategies.mtf.strategy" in sys.modules:
            del sys.modules["titan.strategies.mtf.strategy"]
            
        from titan.strategies.mtf.strategy import MTFConfluenceStrategy, MTFConfluenceConfig
        
        self.MTFConfluenceStrategy = MTFConfluenceStrategy
        self.MTFConfluenceConfig = MTFConfluenceConfig

        # Create dummy config file
        with open("dummy.toml", "wb") as f:
            f.write(b"")

    def tearDown(self):
        self.modules_patcher.stop()
        if Path("dummy.toml").exists():
            Path("dummy.toml").unlink()

    def test_evaluate_confluence_uses_correct_cache_method(self):
        """Test strict mock to ensure cache.position is NOT called with instrument_id."""
        
        config = self.MTFConfluenceConfig(
            instrument_id="EUR/USD",
            bar_types={"H1": "EUR/USD-1h"},
            config_path="dummy.toml"
        )
        # Since StrategyConfig is mocked, the subclass is a Mock. Constructor args don't autoset attrs.
        config.config_path = "dummy.toml"
        config.instrument_id = "EUR/USD"
        config.bar_types = {"H1": "EUR/USD-1h"}
        config.bar_types = {"H1": "EUR/USD-1h"}
        config.warmup_bars = 100
        config.risk_pct = 0.01
        config.leverage_cap = 5.0
        
        strategy = self.MTFConfluenceStrategy(config)
        
        # Setup mocks
        # cache.position(instrument_id) should fail simulation
        # We simulate this by checking call args later, or making it raise if called
        
        def side_effect_position(*args, **kwargs):
            # If argument implies InstrumentId (based on our mock setup), raise TypeError
            # Since we pass strings/mocks, we can just check if anything is passed
            if len(args) > 0:
                 # In the real bug, passing instrument_id here caused TypeError.
                 # So we simulate that.
                 raise TypeError("Argument 'position_id' has incorrect type")
            return None

        strategy.cache.position.side_effect = side_effect_position
        
        # cache.positions SHOULD be called
        # Mocking return value as a list of positions
        mock_pos = MagicMock()
        mock_pos.is_open = True
        mock_pos.side = "LONG"
        strategy.cache.positions.return_value = [mock_pos]
        
        # Manually setup signals to trigger execution logic
        strategy.signals = {"H1": 0.5, "H4": 0.5, "D": 0.5, "W": 0.5} 
        
        # Run _evaluate_confluence
        try:
            strategy._evaluate_confluence(price=Decimal("1.1000"))
        except TypeError as e:
            self.fail(f"Strategy raised TypeError: {e} - Fix did not work!")
            
        # Verify cache.positions was called
        strategy.cache.positions.assert_called()
        
        # Verify cache.position was NOT called (or at least no error raised)
        # Logic: if it was called, side_effect would raise TypeError and fail test.
        
    def test_execute_bias_uses_correct_cache_method(self):
        """Test _execute_bias uses correct cache method."""
        config = self.MTFConfluenceConfig(
            instrument_id="EUR/USD",
            bar_types={"H1": "EUR/USD-1h"},
            config_path="dummy.toml"
        )
        config.config_path = "dummy.toml"
        config.instrument_id = "EUR/USD"
        config.bar_types = {"H1": "EUR/USD-1h"}
        config.risk_pct = 0.01
        config.leverage_cap = 5.0
        
        strategy = self.MTFConfluenceStrategy(config)
        
        # Mock accounts
        strategy.cache.accounts.return_value = {}

        def side_effect_position(*args, **kwargs):
             raise TypeError("Argument 'position_id' has incorrect type")
        strategy.cache.position.side_effect = side_effect_position
        
        strategy.cache.positions.return_value = []
        
        strategy.latest_atr = 0.001
        
        # Run _execute_bias
        try:
            strategy._execute_bias(bias=1, price=Decimal("1.1000"))
        except TypeError as e:
             self.fail(f"Strategy raised TypeError in _execute_bias: {e}")

        # Verify cache.positions called
        strategy.cache.positions.assert_called()

if __name__ == "__main__":
    unittest.main()
