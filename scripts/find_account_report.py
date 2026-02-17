import importlib
import pkgutil

import nautilus_trader


def find_class(package, class_name):
    for _, name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        try:
            module = importlib.import_module(name)
            if hasattr(module, class_name):
                print(f"Found {class_name} in {name}")
                return
        except Exception:
            pass


print("Searching for AccountStatusReport...")
find_class(nautilus_trader, "AccountStatusReport")
print("Search complete.")
