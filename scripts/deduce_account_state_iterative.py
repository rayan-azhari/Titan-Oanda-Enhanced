from nautilus_trader.core.uuid import UUID4
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.events import AccountState
from nautilus_trader.model.identifiers import AccountId
from nautilus_trader.model.objects import AccountBalance, Currency, Money

account_id = AccountId("OANDA-TEST")
account_type = AccountType.MARGIN
base_currency = Currency.from_str("USD")

# Fix AccountBalance (total, free, locked?)
balance = AccountBalance(
    Money(1000, base_currency), Money(1000, base_currency), Money(0, base_currency)
)
ts = 123456789

print("Attempt 1: account_id")
try:
    AccountState(account_id, 2, 3, 4, 5, 6, 7, 8, 9, 10)
except TypeError as e:
    print(f"Error 1: {e}")

print("\nAttempt 2: account_id, account_type")
try:
    AccountState(account_id, account_type, 3, 4, 5, 6, 7, 8, 9, 10)
except TypeError as e:
    print(f"Error 2: {e}")

print("\nAttempt 3: account_id, account_type, base_currency")
try:
    AccountState(account_id, account_type, base_currency, 4, 5, 6, 7, 8, 9, 10)
except TypeError as e:
    print(f"Error 3: {e}")

print("\nAttempt 4: account_id, account_type, base_currency, is_reported")
try:
    # guessing 4th arg is bool is_reported?
    AccountState(account_id, account_type, base_currency, True, 5, 6, 7, 8, 9, 10)
except TypeError as e:
    print(f"Error 4: {e}")


print("\nAttempt 5: all known check + guessing rest")
try:
    # 4th arg is balances (list)
    # 5th: ?
    AccountState(account_id, account_type, base_currency, [balance], 5, 6, 7, 8, 9, 10)
except TypeError as e:
    print(f"Error 5: {e}")

print("\nAttempt 6: guessing 5th is is_reported (bool)")
try:
    AccountState(account_id, account_type, base_currency, [balance], True, 6, 7, 8, 9, 10)
except TypeError as e:
    print(f"Error 6: {e}")

print("\nAttempt 7: guessing 5th is margins (dict) or something else?")
# nautilus often has margins or risk limits
try:
    AccountState(account_id, account_type, base_currency, [balance], {}, 6, 7, 8, 9, 10)
except TypeError as e:
    print(f"Error 7: {e}")


print("\nAttempt 9: guessing 6th is is_reported (bool)")
try:
    # 4th: balances (list)
    # 5th: margins (list)
    # 6th: ?
    AccountState(account_id, account_type, base_currency, [balance], [], True, 7, 8, 9, 10)
except TypeError as e:
    print(f"Error 9: {e}")


print("\nAttempt 11: Hypothesis: 4=bool (is_reported), 5=balances, 6=margins")
try:
    AccountState(account_id, account_type, base_currency, True, [balance], [], 7, 8, 9, 10)
except TypeError as e:
    print(f"Error 11: {e}")

print("\nAttempt 12: Hypothesis: 4=bool, 5=balances, 6=margins, 7=event_id(UUID)")
try:
    AccountState(account_id, account_type, base_currency, True, [balance], [], UUID4(), 8, 9, 10)
except TypeError as e:
    print(f"Error 12: {e}")


print("\nAttempt 14: Hypothesis: 7=info(dict), 8=event_id(UUID)")
try:
    AccountState(account_id, account_type, base_currency, True, [balance], [], {}, UUID4(), 9, 10)
except TypeError as e:
    print(f"Error 14: {e}")


print("\nAttempt 16: Hypothesis: 8=UUID, 9=int, 10=int")
try:
    AccountState(
        account_id,
        account_type,
        base_currency,
        True,
        [balance],
        [],
        {},
        UUID4(),
        123456789,
        123456789,
    )
    print("SUCCESS: AccountState created!")
except TypeError as e:
    print(f"Error 16: {e}")
