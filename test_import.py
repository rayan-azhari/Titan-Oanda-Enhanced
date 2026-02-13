try:
    import nautilus_trader
    with open("import_success.txt", "w") as f:
        f.write(f"Imported nautilus_trader version: {nautilus_trader.__version__}")
except ImportError as e:
    with open("import_failure.txt", "w") as f:
        f.write(str(e))
except Exception as e:
    with open("import_failure.txt", "w") as f:
        f.write(f"Unexpected error: {e}")
