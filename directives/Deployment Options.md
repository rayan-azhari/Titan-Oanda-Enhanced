# Deployment Options: Docker vs. Systemd vs. Tmux

This guide outlines the three primary methods for deploying Titan strategies, comparing their pros, cons, and appropriate use cases. It also provides detailed setup instructions, particularly for **Tmux** on a VPS.

## 📊 Comparison Matrix

| Feature | **Docker** 🐳 | **Systemd** ⚙️ | **Tmux** 🖥️ |
| :--- | :--- | :--- | :--- |
| **Best For** | Production / Cloud Scaling | Production / Single VPS | Testing / Debugging / Short-term |
| **Reliability** | ⭐⭐⭐⭐⭐ (Isolated, Auto-restart) | ⭐⭐⭐⭐⭐ (Native OS Service) | ⭐⭐ (Manual, No auto-restart) |
| **Setup Complexity** | High (Images, Volumes) | Medium (Service files) | Low (Just a command) |
| **Portability** | High (Runs anywhere) | Low (Tied to OS/Paths) | Low (Tied to current session) |
| **Live Monitoring** | Logs only (`docker logs`) | Logs only (`journalctl`) | **Interactive** (See console output) |
| **Auto-Start on Boot** | Yes (`--restart always`) | Yes (`enable`) | No (Manual start required) |

---

## 🏆 The Titan Gold Standard: Best Practice

For a long-term, profitable trading operation, stability is paramount. The recommended architecture for Titan is:

### 1. Infrastructure: Cloud VPS (Compute Engine)
*   **Provider**: Google Cloud Platform (GCE) or AWS EC2.
*   **Region**: `europe-west2` (London) or `us-east1` (New York). **Crucial**: Choose the region closest to OANDA's matching engine (usually New York or London) to minimize latency.
*   **Spec**: 2 vCPU, 4GB RAM (minimum for ML models).

### 2. Deployment Method: Docker 🐳
*   **Why**: Docker guarantees that if it works on your machine, it works on the server. No "it works on my machine" bugs.
*   **Workflow**:
    1.  **Develop Locally**: Test changes -> Commit & Push to GitHub.
    2.  **Deploy on Server**: SSH into VPS -> `git pull` -> `uv run scripts/build_docker.py` -> `docker restart titan`.

### 3. Process Management: Docker Restart Policy
*   Ensure the container always comes back after a crash or reboot:
    ```bash
    docker run --restart always ...
    ```

### 4. Monitoring: The "Guardian" Pattern
*   Do not stare at logs 24/7.
*   Run a lightweight "Guardian" script (or use a tool like UptimeRobot) that pings your strategy's health endpoint or tails logs for "ERROR".
*   If the strategy goes silent for 5 minutes -> **PagerDuty / Slack Alert**.

---

## 1. 🐳 Docker (The Production Standard)

**Why use it?**
Docker ensures the strategy runs in the *exact* same environment on your server as it does on your local machine. It isolates dependencies and prevents conflicts.

**How to use:**
1.  **Build**:
    ```bash
    uv run python scripts/build_docker.py
    ```
2.  **Run**:
    ```bash
    docker run -d \
      --name titan-live \
      --restart always \
      --env-file .env \
      -v $(pwd)/data:/app/data \
      -v $(pwd)/.tmp/logs:/app/.tmp/logs \
      titan-oanda-algo
    ```

---

## 2. ⚙️ Systemd (The VPS Standard)

**Why use it?**
Systemd is built into Linux. It treats your strategy like a background service (like a web server). If the strategy crashes or the server reboots, Systemd automatically restarts it.

**How to use:**
Create `/etc/systemd/system/titan.service`:
```ini
[Unit]
Description=Titan Strategy
After=network.target

[Service]
User=root
WorkingDirectory=/root/Titan-Oanda
ExecStart=/root/.local/bin/uv run python scripts/run_live_mtf.py
Restart=always
RestartSec=5
EnvironmentFile=/root/Titan-Oanda/.env

[Install]
WantedBy=multi-user.target
```

---

## 3. 🖥️ Tmux (Interactive / Testing)

**Why use it?**
Tmux allows you to detach from a terminal session without killing the processes running inside it.

**The "Local PC vs. VPS" Question**
*   **Can I run it on my Local PC?** Yes, but **not recommended** for live trading.
    *   *Risks:* Power outages, internet disconnects, Windows updates rebooting your PC.
    *   *Use Case:* Development testing only.
*   **Do I need a VPS?** **YES.**
    *   *Why:* A VPS (Virtual Private Server) runs 24/7 in a datacenter with redundant power and internet.
    *   *Cost:* ~$5-10/mo (DigitalOcean, Vultr, Hetzner).
    *   *Latency:* VPS servers are often physically closer to OANDA's servers (NY/London), executing trades faster than your home PC.

### 🛑 Detailed Tmux Workflow on VPS

1.  **Connect to VPS**:
    ```bash
    ssh user@your-vps-ip
    ```
2.  **Start a New Session**:
    ```bash
    tmux new -s titan
    ```
    *You are now inside a "virtual" terminal.*
3.  **Run the Strategy**:
    ```bash
    cd Titan-Oanda
    uv run python scripts/run_live_mtf.py
    ```
    *You will see the strategy logs and dashboard updating in real-time.*
4.  **Detach (Leave it running)**:
    *   Press `Ctrl` + `B`
    *   Release both keys.
    *   Press `D`.
    *   *You are now back in your main SSH session. The strategy is still running in the background.*
5.  **Disconnect from VPS**:
    ```bash
    exit
    ```
    *You can now turn off your computer. The VPS and Strategy keep running.*

6.  **Re-Check (Monitoring)**:
    *   SSH back into the VPS.
    *   Attach to the session:
        ```bash
        tmux attach -t titan
        ```
    *   *You are back in the "virtual" terminal, seeing the live logs exactly as you left them.*

### ⚠️ Critical Warning for Tmux
If the server reboots (maintenance) or the python script crashes, **Tmux will NOT restart it automatically**.
*   **Recommendation**: Use Tmux for the first 24-48 hours to monitor a new strategy. Once stable, switch to **Systemd** or **Docker** for long-term peace of mind.
