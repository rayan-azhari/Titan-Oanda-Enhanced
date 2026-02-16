"""setup_env.py — Interactive environment configuration for the workspace.

Guides the user through setting up their .env file with OANDA credentials
and optional Slack webhook URL.

Directive: Workspace Initialisation.md
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ENV_FILE = PROJECT_ROOT / ".env"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"


def main() -> None:
    """Guide the user through environment setup."""
    print("=" * 50)
    print("  Titan-Oanda-Algo — Environment Setup")
    print("=" * 50)
    print()

    if ENV_FILE.exists():
        print(f"  ⚠️  {ENV_FILE} already exists.")
        overwrite = input("  Overwrite? (y/N): ").strip().lower()
        if overwrite != "y":
            print("  Keeping existing .env. Done.")
            return

    print("  Enter your OANDA API credentials:\n")

    account_id = input("  OANDA Account ID: ").strip()
    access_token = input("  OANDA Access Token: ").strip()
    environment = input("  Environment (practice/live) [practice]: ").strip() or "practice"
    slack_url = input("  Slack Webhook URL (optional, press Enter to skip): ").strip()

    # Write .env file
    lines = [
        "# ──────────────────────────────────────────────────────────",
        "# OANDA API Credentials",
        "# ──────────────────────────────────────────────────────────",
        f"OANDA_ACCOUNT_ID={account_id}",
        f"OANDA_ACCESS_TOKEN={access_token}",
        f"OANDA_ENVIRONMENT={environment}",
        "",
    ]

    if slack_url:
        lines.extend(
            [
                "# ──────────────────────────────────────────────────────────",
                "# Slack Notifications (Guardian Agent)",
                "# ──────────────────────────────────────────────────────────",
                f"SLACK_WEBHOOK_URL={slack_url}",
                "",
            ]
        )

    ENV_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  ✓ Created {ENV_FILE}")
    print("  Verify with: uv run python scripts/verify_connection.py")
    print("\n✅ Environment setup complete.\n")


if __name__ == "__main__":
    main()
