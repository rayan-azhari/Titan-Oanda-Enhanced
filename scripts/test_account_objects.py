from nautilus_trader.execution.reports import ExecutionMassStatus
from nautilus_trader.model.events import AccountState

print("--- AccountState ---")
try:
    AccountState()
except TypeError as e:
    print(f"Error: {e}")

print("\n--- ExecutionMassStatus ---")
print([x for x in dir(ExecutionMassStatus) if "add" in x])
