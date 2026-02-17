"""State manager for MTF Optimization Workflow.

Handles persistence of optimization results between stages using a JSON file.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Path to the shared state file
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATE_FILE = PROJECT_ROOT / ".tmp" / "mtf_state.json"


def load_state() -> Dict[str, Any]:
    """Load the current optimization state."""
    if not STATE_FILE.exists():
        return {}
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def save_state(updates: Dict[str, Any]) -> None:
    """Update and save the optimization state."""
    current = load_state()
    current.update(updates)
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(current, f, indent=4)
    print(f"[State] Updated {STATE_FILE.name}: {list(updates.keys())}")


def save_stage1(ma_type: str, threshold: float) -> None:
    """Save Stage 1 results (Global Params)."""
    save_state({"stage1": {"ma_type": ma_type, "threshold": threshold}})


def get_stage1() -> Optional[Tuple[str, float]]:
    """Retrieve Stage 1 results. Returns (ma_type, threshold) or None."""
    state = load_state()
    s1 = state.get("stage1")
    if s1:
        return s1.get("ma_type"), s1.get("threshold")
    return None


def save_stage2(weights: Dict[str, float]) -> None:
    """Save Stage 2 results (Weights)."""
    save_state({"stage2": {"weights": weights}})


def get_stage2() -> Optional[Dict[str, float]]:
    """Retrieve Stage 2 results (Weights)."""
    state = load_state()
    s2 = state.get("stage2")
    return s2.get("weights") if s2 else None


def save_stage3(params: Dict[str, Any]) -> None:
    """Save Stage 3 results (Indicator Params)."""
    save_state({"stage3": {"params": params}})
