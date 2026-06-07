#!/usr/bin/env python3
"""
Start all Flask web tools (Web tools hub, ROI editor, ROI previewer, pipeline debug,
human review, XLSX compare, XLSX mapping editor, ROI-to-XLSX visualizer).

  # Background daemons (current terminal returns); logs under testing/output/web_tools/
  python3 tools/start_web_tools.py

  # Open a new terminal window that runs the servers (Ctrl+C stops all)
  python3 tools/start_web_tools.py --terminal

  # Run all servers in this terminal (foreground; Ctrl+C stops all)
  python3 tools/start_web_tools.py --watch

  # Stop daemons started earlier (uses PID file)
  python3 tools/start_web_tools.py --stop

  # Start without opening a browser tab (e.g. headless/CI)
  python3 tools/start_web_tools.py --no-browser

By default the Web tools hub (http://127.0.0.1:4999/) opens in a new browser tab once the port is ready.

Requires: Flask, same environment as the rest of the project. Run from repo root or any cwd.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any

# Repo root (parent of tools/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "testing" / "output" / "web_tools"
PID_FILE = LOG_DIR / "web_tools_pids.json"

# (label, tools/<script>.py, default port)
WEB_TOOLS: tuple[tuple[str, str, int], ...] = (
    # First entry is the "main" page opened in the browser.
    ("hub", "hub_web.py", 4999),
    ("roi_editor", "roi_editor_web.py", 5000),
    ("roi_previewer", "roi_previewer_web.py", 5001),
    ("pipeline_debug", "pipeline_debug_web.py", 5002),
    ("human_review", "human_review_web.py", 5003),
    ("xlsx_compare", "xlsx_compare_web.py", 5004),
    ("xlsx_mapping", "xlsx_mapping_web.py", 5005),
    ("pipeline_map", "pipeline_map_web.py", 5006),
)
LEGACY_WEB_TOOL_SCRIPTS: tuple[str, ...] = (
    "evms_hub_web.py",
)


def _tool_path(script: str) -> Path:
    return PROJECT_ROOT / "tools" / script


def _hub_url() -> str:
    _label, _script, port = WEB_TOOLS[0]
    return f"http://127.0.0.1:{port}/"


def _wait_tcp_open(host: str, port: int, timeout_s: float, interval_s: float = 0.12) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.35):
                return True
        except OSError:
            time.sleep(interval_s)
    return False


def open_hub_in_browser(*, block: bool, timeout_s: float = 25.0) -> None:
    """Open Web tools hub in a new browser tab once the port accepts connections."""

    def _run() -> None:
        url = _hub_url()
        if not _wait_tcp_open("127.0.0.1", WEB_TOOLS[0][2], timeout_s):
            print(f"Hub did not become ready in {timeout_s}s; open manually: {url}", file=sys.stderr)
            return
        try:
            webbrowser.open(url, new=2)
        except Exception as e:  # pragma: no cover - browser/env specific
            print(f"Could not open browser ({e}). Open manually: {url}", file=sys.stderr)

    if block:
        _run()
    else:
        t = threading.Thread(target=_run, name="hub-open-browser", daemon=True)
        t.start()


def _ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _managed_script_markers() -> tuple[str, ...]:
    scripts = {script for _label, script, _port in WEB_TOOLS}
    scripts.update(LEGACY_WEB_TOOL_SCRIPTS)
    return tuple(f"/tools/{script}" for script in sorted(scripts))


def _kill_stale_web_tool_processes(*, include_current_run: bool = True) -> int:
    """
    Best-effort cleanup for stale web tool processes from previous runs.
    This avoids legacy servers (e.g. evms_hub_web.py) holding ports and serving old assets.
    """
    if os.name != "posix":
        return 0

    markers = _managed_script_markers()
    me = os.getpid()
    victims: list[int] = []

    try:
        out = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
    except Exception:
        return 0

    for raw in out.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        args = parts[1]
        if pid == me:
            continue
        if not include_current_run and "start_web_tools.py" in args:
            continue
        if any(marker in args for marker in markers):
            victims.append(pid)

    if not victims:
        return 0

    signaled = 0
    for pid in victims:
        try:
            os.kill(pid, signal.SIGTERM)
            signaled += 1
            print(f"  cleanup: SIGTERM pid={pid}")
        except ProcessLookupError:
            pass
        except OSError:
            pass

    time.sleep(0.35)
    for pid in victims:
        try:
            os.kill(pid, 0)
            os.kill(pid, signal.SIGKILL)
            print(f"  cleanup: SIGKILL pid={pid}")
        except ProcessLookupError:
            pass
        except OSError:
            pass
    return signaled


def _popen_server(label: str, script: str, port: int) -> subprocess.Popen[bytes]:
    log_path = LOG_DIR / f"{label}.log"
    f = open(log_path, "ab", buffering=0)  # noqa: SIM115
    cmd = [
        sys.executable,
        str(_tool_path(script)),
        "--port",
        str(port),
    ]
    return subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdin=subprocess.DEVNULL,
        stdout=f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )


def start_daemon(*, open_browser: bool = True) -> list[dict[str, Any]]:
    """Start each tool in the background; return metadata for PID file."""
    _ensure_log_dir()
    _kill_stale_web_tool_processes()
    records: list[dict[str, Any]] = []
    for label, script, port in WEB_TOOLS:
        if not _tool_path(script).is_file():
            print(f"Warning: missing {script}, skipping {label}", file=sys.stderr)
            continue
        proc = _popen_server(label, script, port)
        records.append(
            {
                "label": label,
                "script": script,
                "port": port,
                "pid": proc.pid,
                "log": str(LOG_DIR / f"{label}.log"),
            }
        )
        print(f"  [{label}] pid={proc.pid}  http://127.0.0.1:{port}/  log={LOG_DIR / f'{label}.log'}")
    if records:
        PID_FILE.write_text(json.dumps(records, indent=2), encoding="utf-8")
        if open_browser:
            open_hub_in_browser(block=True)
    return records


def stop_daemon() -> int:
    """Stop processes listed in PID file. Returns number of signals sent."""
    cleaned = _kill_stale_web_tool_processes()
    if not PID_FILE.is_file():
        if cleaned:
            print(f"Stopped {cleaned} process(es) via cleanup.")
            return 0
        print(f"No PID file at {PID_FILE} (nothing to stop).", file=sys.stderr)
        return 1
    try:
        records = json.loads(PID_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"Could not read {PID_FILE}: {e}", file=sys.stderr)
        return 1
    n = 0
    for r in records:
        pid = int(r.get("pid", 0))
        if pid <= 0:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            n += 1
            print(f"  SIGTERM pid={pid} ({r.get('label', '?')})")
        except ProcessLookupError:
            print(f"  (pid {pid} already gone)")
        except OSError as e:
            print(f"  pid {pid}: {e}", file=sys.stderr)
    time.sleep(0.5)
    for r in records:
        pid = int(r.get("pid", 0))
        if pid <= 0:
            continue
        try:
            os.kill(pid, 0)
            os.kill(pid, signal.SIGKILL)
            print(f"  SIGKILL pid={pid}")
        except ProcessLookupError:
            pass
        except OSError:
            pass
    try:
        PID_FILE.unlink()
    except OSError:
        pass
    total = n + cleaned
    print(f"Stopped {total} process(es).")
    return 0 if total else 1


def run_watch(*, open_browser: bool = True) -> None:
    """Foreground: start all servers, wait until Ctrl+C, then terminate children."""
    _ensure_log_dir()
    _kill_stale_web_tool_processes()
    procs: list[tuple[str, subprocess.Popen[bytes]]] = []
    try:
        for label, script, port in WEB_TOOLS:
            if not _tool_path(script).is_file():
                print(f"Warning: missing {script}, skipping {label}", file=sys.stderr)
                continue
            log_path = LOG_DIR / f"{label}.log"
            f = open(log_path, "ab", buffering=0)  # noqa: SIM115
            cmd = [
                sys.executable,
                str(_tool_path(script)),
                "--port",
                str(port),
            ]
            p = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdin=subprocess.DEVNULL,
                stdout=f,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            procs.append((label, p))
            print(f"  [{label}] pid={p.pid}  http://127.0.0.1:{port}/  log={log_path}")

        if not procs:
            print("No servers started.", file=sys.stderr)
            return

        if open_browser:
            open_hub_in_browser(block=False)

        print("\nRunning (Ctrl+C to stop all). Logs:", LOG_DIR)
        while True:
            time.sleep(1)
            for label, p in procs:
                if p.poll() is not None:
                    print(f"  [{label}] exited with code {p.returncode}", file=sys.stderr)
            if all(p.poll() is not None for _, p in procs):
                break
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        for label, p in procs:
            if p.poll() is None:
                try:
                    p.send_signal(signal.SIGTERM)
                except OSError:
                    pass
        time.sleep(0.3)
        for label, p in procs:
            if p.poll() is None:
                try:
                    p.kill()
                except OSError:
                    pass
        print("Stopped.")


def _try_open_terminal(*, no_browser: bool = False) -> bool:
    """Launch a new terminal running this script in --watch mode."""
    from shutil import which

    root = str(PROJECT_ROOT)
    script = str(Path(__file__).resolve())
    nb = " --no-browser" if no_browser else ""
    inner = f"cd {shlex.quote(root)} && exec {shlex.quote(sys.executable)} {shlex.quote(script)} --watch{nb}"
    # Keep shell open after exit so user can read messages
    pause = f"{inner}; echo; read -r -p 'Press Enter to close…' _"

    candidates: list[list[str]] = []

    for spec in (
        ["gnome-terminal", "--", "bash", "-lc", pause],
        ["konsole", "-e", f"bash -lc {shlex.quote(pause)}"],
        ["xfce4-terminal", "-e", f"bash -lc {shlex.quote(pause)}"],
        ["alacritty", "-e", "bash", "-lc", pause],
        ["kitty", "bash", "-lc", pause],
        ["x-terminal-emulator", "-e", f"bash -lc {shlex.quote(pause)}"],
    ):
        candidates.append(spec)

    for argv in candidates:
        exe = argv[0]
        if not exe.startswith("/") and os.path.dirname(exe) == "":
            if not which(exe):
                continue
        try:
            subprocess.Popen(argv, cwd=root, start_new_session=True)
            print(f"Started new terminal ({exe}).")
            return True
        except OSError:
            continue
    return False


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Start or stop all Web tools.")
    p.add_argument(
        "--terminal",
        action="store_true",
        help="Open a new graphical terminal and run servers there (--watch)",
    )
    p.add_argument(
        "--watch",
        action="store_true",
        help="Run all servers in this process; block until Ctrl+C (used by --terminal)",
    )
    p.add_argument(
        "--stop",
        action="store_true",
        help="Stop background servers using the saved PID file",
    )
    p.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the Web tools hub in a new browser tab",
    )
    args = p.parse_args()

    if args.stop:
        return stop_daemon()

    open_browser = not args.no_browser

    if args.watch:
        run_watch(open_browser=open_browser)
        return 0

    if args.terminal:
        if not _try_open_terminal(no_browser=args.no_browser):
            print(
                "Could not open a new terminal. Starting in background in this session instead.\n"
                f"Logs: {LOG_DIR}\n"
                "Or run: python3 tools/start_web_tools.py --watch",
                file=sys.stderr,
            )
            start_daemon(open_browser=open_browser)
        return 0

    print(
        f"Starting web tools in background…\nLogs: {LOG_DIR}\n"
        "Stop: python3 tools/start_web_tools.py --stop\n"
    )
    start_daemon(open_browser=open_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
