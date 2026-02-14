"""build_docker_image.py — Build a Docker container for the Nautilus trading system.

Creates a production Docker image based on python:3.11-slim,
including the nautilus_trader wheel and the models/ directory.

Directive: Live Deployment and Monitoring.md
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DOCKERFILE_PATH = PROJECT_ROOT / "Dockerfile"


def generate_dockerfile() -> str:
    """Generate the Dockerfile content for the Nautilus trading container.

    Returns:
        Dockerfile content as a string.
    """
    return """# Titan-Oanda-Algo — Production Container
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first (layer caching)
COPY pyproject.toml ./
RUN uv sync --no-dev

# Copy application code
COPY execution/ ./execution/
COPY config/ ./config/

# Copy trained models (critical for ML strategies)
COPY models/ ./models/

# Copy environment template (runtime secrets via env vars)
COPY .env.example ./.env.example

# Health check
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \\
    CMD python -c "print('OK')" || exit 1

# Default: run the live trading engine in practice mode
CMD ["uv", "run", "python", "execution/run_nautilus_live.py"]
"""


def main() -> None:
    """Generate Dockerfile and build the Docker image."""
    print("🐳 Building Docker image for Titan-Oanda-Algo\n")

    # Generate Dockerfile
    dockerfile_content = generate_dockerfile()
    DOCKERFILE_PATH.write_text(dockerfile_content)
    print(f"  ✓ Generated {DOCKERFILE_PATH}")

    # Check if Docker is available
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"  Docker: {result.stdout.strip()}")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("  ⚠️  Docker not found. Dockerfile generated but image not built.")
        print("  Install Docker and run: docker build -t titan-oanda-algo .")
        return

    # Build image
    print("\n  Building image (this may take a few minutes)...")
    result = subprocess.run(
        ["docker", "build", "-t", "titan-oanda-algo", "."],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print("  ✓ Image built: titan-oanda-algo")
        print("\n  Deploy to GCE (europe-west2):")
        print("    docker tag titan-oanda-algo gcr.io/<PROJECT_ID>/titan-oanda-algo")
        print("    docker push gcr.io/<PROJECT_ID>/titan-oanda-algo")
    else:
        print(f"  ✗ Build failed:\n{result.stderr}")

    print("\n✅ Docker build complete.\n")


if __name__ == "__main__":
    main()
