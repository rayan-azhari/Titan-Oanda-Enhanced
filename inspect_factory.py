import inspect

from nautilus_trader.common.factories import OrderFactory
from nautilus_trader.model.orders import MarketIfTouchedOrder, StopMarketOrder

print("--- OrderFactory ---")
print([m for m in dir(OrderFactory) if not m.startswith("_")])

print("\n--- StopMarketOrder Arguments ---")
try:
    print(inspect.signature(StopMarketOrder.__init__))
except Exception as e:
    print(f"Failed to inspect signature: {e}")
    # Inspect docstring?
    print(StopMarketOrder.__init__.__doc__)

print("\n--- MarketIfTouchedOrder Arguments ---")
try:
    print(inspect.signature(MarketIfTouchedOrder.__init__))
except Exception as e:
    print(f"Failed to inspect signature: {e}")
