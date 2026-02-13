
import sys
import platform
import os

try:
    import nautilus_trader
    nautilus_version = nautilus_trader.__version__
    status = "SUCCESS"
except ImportError as e:
    nautilus_version = str(e)
    status = "FAILURE"

with open("env_check_result.txt", "w") as f:
    f.write(f"Status: {status}\n")
    f.write(f"Python: {sys.version}\n")
    f.write(f"Platform: {platform.platform()}\n")
    f.write(f"CWD: {os.getcwd()}\n")
    f.write(f"NautilusTrader: {nautilus_version}\n")
