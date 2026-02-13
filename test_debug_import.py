
import sys
import traceback
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    print("Attempting to import execution.nautilus_oanda.instruments...")
    from execution.nautilus_oanda.instruments import OandaInstrumentProvider
    
    with open("debug_import_success.txt", "w") as f:
        f.write("Successfully imported OandaInstrumentProvider")
        
except Exception:
    with open("debug_import_failure.txt", "w") as f:
        f.write(traceback.format_exc())
