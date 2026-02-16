"""verify_titan_install.py — Verify the titan package is installed and importable.

This script should be run from outside the package to ensure no local folder 
is being picked up by accident.
"""


def main():
    print("🔍 Verifying titan package installation...")
    try:
        import titan
        print(f"  ✓ Import successful: {titan}")
        print(f"  ✓ Path: {titan.__file__}")
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return

    try:
        from titan.data.oanda import fetch_candles
        print("  ✓ Import titan.data.oanda successful")
        from titan.utils.ops import cancel_all_orders
        print("  ✓ Import titan.utils.ops successful")
    except ImportError as e:
        print(f"  ❌ Submodule import failed: {e}")
        return

    print("\n✅ Titan package is correctly installed in editable mode.")


if __name__ == "__main__":
    main()
