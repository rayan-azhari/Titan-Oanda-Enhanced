"""validate_data.py — Validate raw OHLCV Parquet files.

Moved from titan/data/validation.py.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from titan.data.validation import check_duplicates, check_gaps, check_outliers, check_negative_volume

RAW_DATA_DIR = PROJECT_ROOT / "data"

def main() -> None:
    """Validate all raw Parquet files in data/."""
    raw_files = sorted(RAW_DATA_DIR.glob("*.parquet"))
    if not raw_files:
        print(f"ERROR: No raw data in {RAW_DATA_DIR}. Run download_data.py first.")
        sys.exit(1)

    print(f"🔍 Validating {len(raw_files)} dataset(s)\n")

    total_issues = 0
    for path in raw_files:
        name = path.stem
        print(f"━━━ {name} ━━━")
        df = pd.read_parquet(path)

        # Infer frequency from filename
        if "M5" in name:
            freq = "5min"
        elif "M1" in name:
            freq = "1min"
        elif "H1" in name:
            freq = "1h"
        elif "H4" in name:
            freq = "4h"
        else:
            freq = "5min"

        total_issues += check_duplicates(df, name)
        total_issues += check_gaps(df, name, expected_freq=freq)
        total_issues += check_outliers(df, name)
        total_issues += check_negative_volume(df, name)
        print()

    if total_issues > 0:
        print(f"⚠️  Total issues found: {total_issues}")
    else:
        print("✅ All datasets passed validation.\n")


if __name__ == "__main__":
    main()
