"""send_notification.py — Send Slack alerts for the Guardian monitoring agent.

Sends formatted notifications to a Slack webhook URL when errors
are detected in the live trading system logs.

Directive: Live Deployment and Monitoring.md
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")


def send_slack_message(message: str, severity: str = "warning") -> bool:
    """Send a message to the configured Slack webhook.

    Args:
        message: The alert message body.
        severity: Alert severity — "info", "warning", or "critical".

    Returns:
        True if message was sent successfully, False otherwise.
    """
    if not SLACK_WEBHOOK_URL:
        print("WARNING: SLACK_WEBHOOK_URL not set in .env. Notification skipped.")
        return False

    emoji_map = {
        "info": "ℹ️",
        "warning": "⚠️",
        "critical": "🚨",
    }
    emoji = emoji_map.get(severity, "📢")

    payload = {
        "text": f"{emoji} *Titan-Oanda-Algo Alert*\n*Severity:* {severity.upper()}\n\n{message}",
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        SLACK_WEBHOOK_URL,
        data=data,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                print(f"  ✓ Slack notification sent ({severity})")
                return True
            else:
                print(f"  ✗ Slack returned status {response.status}")
                return False
    except Exception as e:
        print(f"  ✗ Failed to send Slack notification: {e}")
        return False


def main() -> None:
    """Send a test notification or process command-line arguments."""
    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
        severity = "warning"
    else:
        message = "This is a test notification from the Titan-Oanda-Algo Guardian agent."
        severity = "info"

    print("📢 Sending Slack notification...\n")
    send_slack_message(message, severity=severity)
    print("\n✅ Done.\n")


if __name__ == "__main__":
    main()
