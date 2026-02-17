from nautilus_trader.model.events import AccountState

try:
    AccountState(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
except TypeError as e:
    print(f"Error: {e}")
