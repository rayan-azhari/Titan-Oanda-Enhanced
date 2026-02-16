try:
    import nautilus_trader.core.datetime as dt

    print("Found nautilus_trader.core.datetime")
    print(dir(dt))
except ImportError:
    print("nautilus_trader.core.datetime not found")

try:
    from nautilus_trader.model.identifiers import TraderId  # noqa: F401

    print("Found TraderId")
except ImportError:
    print("TraderId import failed")
