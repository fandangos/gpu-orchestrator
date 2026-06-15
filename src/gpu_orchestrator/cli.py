"""gpurun — GPU job supervisor CLI.

Usage:
  gpurun [--auto] [--no-restart] [--] CMD [ARGS...]   run command with GPU mgmt
  gpurun [--auto] [--no-restart] -c 'STRING'           run compound command
  gpurun status                                        show desired/actual/health
  gpurun log [-f] [N]                                  view logs
  gpurun on | off                                      enable/disable server mgmt
  gpurun use NAME|PATH                                 switch model script
  gpurun guard                                         cron self-heal entry
  gpurun detect                                        show detected environment
  gpurun setup                                         interactive setup wizard
  gpurun __match 'CMD'                                 pattern test (hook API)

Exit codes (infra failures only; command rc passes through):
  95  llama-server would not die (needs human)
  96  GPU lock held by another gpurun
  97  llama-server failed to spawn on restore
  98  llama-server not healthy within timeout after restore
  99  usage / config error
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from .config import Config, load
from .health import health_ok, models_ok
from .lifecycle import ProcessDriver
from .logging import EventLogger
from .patterns import matches_gpu

import psutil


# -------------------------------------------------------------------- helpers
def _state_dir() -> Path:
    return Path.home() / ".local" / "state" / "gpu-orchestrator"


def _now_ms() -> int:
    return int(time.time() * 1000)


def _active_conns(port: int) -> int:
    """Count established TCP connections to `port`."""
    try:
        count = 0
        for conn in psutil.net_connections(kind="inet"):
            if (
                conn.laddr
                and conn.laddr.port == port
                and conn.status == psutil.CONN_ESTABLISHED
            ):
                count += 1
        return count
    except (psutil.AccessDenied, OSError):
        return 0


# --------------------------------------------------------- command: run
def cmd_run(cfg: Config, args: argparse.Namespace) -> int:
    """Execute a command with GPU arbitration."""
    auto = args.auto
    no_restart = args.no_restart
    cmd_str = args.command_string
    cmd_argv = args.argv if not cmd_str else None

    if cmd_str:
        cmd = ["bash", "-c", cmd_str]
        shown = cmd_str
    else:
        if not cmd_argv:
            print("[gpurun] no command given", file=sys.stderr)
            return 99
        cmd = cmd_argv
        shown = " ".join(cmd_argv)

    # Auto mode: skip GPU mgmt if no pattern match
    if auto:
        pattern = matches_gpu(shown, cfg.extra_patterns)
        if not pattern:
            result = subprocess.run(cmd)
            sys.exit(result.returncode)

    # Acquire GPU lock
    state_dir = cfg.state_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    lockfile = state_dir / "gpu.lock"

    lock_fd = None
    try:
        lock_fd = os.open(str(lockfile), os.O_CREAT | os.O_WRONLY)
        # Try immediate lock; if held, do a blocking wait with timeout
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        # Lock is held — wait up to cfg.lock_timeout seconds
        deadline = time.monotonic() + cfg.lock_timeout
        while time.monotonic() < deadline:
            time.sleep(0.5)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                continue
        else:
            print(f"[gpurun] GPU lock timeout after {cfg.lock_timeout}s", file=sys.stderr)
            os.close(lock_fd)
            return 96
    except Exception:
        if lock_fd is not None:
            os.close(lock_fd)
        return 96

    run_id = f"run-{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"
    logger = EventLogger(state_dir, run_id)
    run_log = state_dir / "runs" / f"{run_id}.log"
    run_log.parent.mkdir(parents=True, exist_ok=True)

    logger.evt("run_requested", cmd=shown[:300])

    # Build driver
    driver = ProcessDriver(
        start_cmd=cfg.llama_start_cmd or "",
        proc_pattern=cfg.llama_proc_pattern,
        health_url=cfg.llama_url,
        port=cfg.llama_port,
        state_dir=state_dir,
        stop_timeout=cfg.stop_timeout,
    )

    was_running = driver.is_running()
    restore_needed = was_running and not no_restart

    if was_running:
        # Grace period: wait for active connections to drain
        conns = _active_conns(cfg.llama_port)
        if conns > 0:
            logger.evt("grace_wait", conns=conns, max_s=cfg.grace_timeout)
            waited = 0
            while waited < cfg.grace_timeout:
                time.sleep(3)
                waited += 3
                conns = _active_conns(cfg.llama_port)
                if conns == 0:
                    break
            if conns > 0:
                logger.evt("grace_proceed", conns_still=conns)

        logger.evt("llama_stopping")
        if not driver.stop():
            logger.evt("llama_stop_failed")
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
            print(
                "[gpurun] llama-server did not terminate; GPU job NOT run. "
                "Inspect: ps aux | grep llama",
                file=sys.stderr,
            )
            return 95

        logger.evt("llama_stopped", ms=_now_ms())
        # Wait for CUDA page-locked memory to actually release (not just process gone).
        print("[gpurun] waiting for CUDA context teardown...", file=sys.stderr)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            time.sleep(1)
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n"):
                        line = line.strip()
                        mem_util = line.split(",")
                        if len(mem_util) >= 2:
                            mem = float(mem_util[0].strip())
                            util = float(mem_util[1].strip())
                            # If GPU is idle (<5% util) and uses <100MB, it's clean
                            if util < 5 and mem < 100:
                                continue
                            else:
                                break
                    else:
                        break  # all GPUs clean
            except (subprocess.TimeoutExpired, ValueError):
                pass

    cmd_timeout = args.timeout if args.timeout is not None else cfg.command_timeout
    t_all = _now_ms()
    t_cmd = _now_ms()
    rc = 0
    restore_done = False

    try:
        with open(run_log, "wb") as run_fh:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )

            def tee(stream, fh, out_stream):
                for line in iter(stream.readline, b""):
                    fh.write(line)
                    fh.flush()
                    if out_stream is not None:
                        out_stream.buffer.write(line)
                        out_stream.buffer.flush()

            t1 = threading.Thread(target=tee, args=(proc.stdout, run_fh, sys.stdout), daemon=True)
            t2 = threading.Thread(target=tee, args=(proc.stderr, run_fh, sys.stderr), daemon=True)

            # Watchdog: if gpurun is killed (SIGHUP from tmux death),
            # kill the child process so CUDA context frees up.
            def watchdog():
                while proc.poll() is None:
                    time.sleep(1)
                try:
                    parent = psutil.Process(proc.pid)
                    for child in parent.children(recursive=True):
                        child.kill()
                    parent.kill()
                except (psutil.NoSuchProcess, OSError):
                    pass

            wd = threading.Thread(target=watchdog, daemon=True)
            t1.start()
            t2.start()
            wd.start()
            rc = proc.wait(cmd_timeout)
            if rc is None:
                logger.evt("cmd_timeout", timeout_s=cmd_timeout)
                print(f"[gpurun] command timed out after {cmd_timeout}s, killing...", file=sys.stderr)
                try:
                    parent = psutil.Process(proc.pid)
                    for child in parent.children(recursive=True):
                        child.kill()
                    parent.kill()
                except psutil.NoSuchProcess:
                    pass
                rc = -9

            # Close pipes so tee threads get EOF and exit
            try:
                proc.stdout.close()
            except Exception:
                pass
            try:
                proc.stderr.close()
            except Exception:
                pass
            t1.join(timeout=5)
            t2.join(timeout=5)

            # Wait for CUDA context to free (page-locked memory lingers after process death).
            # Without this, restore() starts llama-server while VRAM is still hoarded by zombies.
            if was_running:
                print("[gpurun] waiting for CUDA context teardown...", file=sys.stderr)
                for _ in range(30):  # up to 15s
                    time.sleep(0.5)
                    # Check if any GPU memory is still > idle level (~34MB baseline per card)
                    try:
                        result = subprocess.run(
                            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=5,
                        )
                        if result.returncode == 0:
                            total_used = sum(float(m.strip()) for m in result.stdout.strip().split("\n") if m.strip())
                            if total_used < 200:  # ~100MB per GPU = clean
                                break
                    except (subprocess.TimeoutExpired, ValueError):
                        pass
    except Exception as e:
        print(f"[gpurun] command failed: {e}", file=sys.stderr)
        rc = 1

    logger.evt("cmd_exit", rc=rc, ms=_now_ms() - t_cmd)

    # Restore server
    if restore_needed:
        restore_done = True
        rrc, code = driver.restore(cfg.health_timeout)
        if not rrc:
            logger.evt(
                "run_complete",
                rc=code,
                cmd_rc=rc,
                ms=_now_ms() - t_all,
            )
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
            print(
                f"[gpurun] command rc={rc}, but llama-server restore FAILED (code={code}). "
                f"Try: gpurun on",
                file=sys.stderr,
            )
            return code

    logger.evt("run_complete", rc=rc, ms=_now_ms() - t_all)

    fcntl.flock(lock_fd, fcntl.LOCK_UN)
    os.close(lock_fd)

    if restore_needed:
        print(
            f"[gpurun] command finished rc={rc}; "
            f"restoring llama-server (output: {run_log})",
            file=sys.stderr,
        )
    else:
        status = "not running" if not was_running else ("left as-is" if no_restart else "restored")
        print(
            f"[gpurun] command finished rc={rc} (llama-server was {status}). "
            f"Output: {run_log}",
            file=sys.stderr,
        )

    return rc


# ------------------------------------------------------- command: guard
def cmd_guard(cfg: Config, args: argparse.Namespace) -> int:
    """Cron self-heal: restore server if it's down and desired=on."""
    desired_file = cfg.state_dir / "desired"
    try:
        desired = desired_file.read_text().strip()
    except OSError:
        desired = "on"

    if desired != "on":
        return 0

    # Don't run if a gpurun job is active
    lockfile = cfg.state_dir / "gpu.lock"
    try:
        lock_fd = os.open(str(lockfile), os.O_CREAT | os.O_WRONLY)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Got the lock → no other gpurun is active
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    except OSError:
        return 0  # another gpurun is active

    logger = EventLogger(cfg.state_dir, "guard")
    driver = ProcessDriver(
        start_cmd=cfg.llama_start_cmd or "",
        proc_pattern=cfg.llama_proc_pattern,
        health_url=cfg.llama_url,
        port=cfg.llama_port,
        state_dir=cfg.state_dir,
        stop_timeout=cfg.stop_timeout,
    )

    if driver.is_running():
        return 0

    logger.evt("guard_restoring")
    success, code = driver.restore(cfg.health_timeout)
    return code if not success else 0


# ------------------------------------------------------ command: status
def cmd_status(cfg: Config, args: argparse.Namespace) -> int:
    """Show current state."""
    driver = ProcessDriver(
        start_cmd=cfg.llama_start_cmd or "",
        proc_pattern=cfg.llama_proc_pattern,
        health_url=cfg.llama_url,
        port=cfg.llama_port,
        state_dir=cfg.state_dir,
        stop_timeout=cfg.stop_timeout,
    )

    print(f"driver    : process")

    desired = "on"
    desired_file = cfg.state_dir / "desired"
    if desired_file.exists():
        desired = desired_file.read_text().strip()
    print(f"desired   : {desired}")

    model_link = Path.home() / ".config" / "gpu-orchestrator" / "current-model.sh"
    if model_link.exists():
        print(f"model     : {model_link.resolve()}")
    else:
        print("model     : UNSET")

    if driver.is_running():
        pid = driver.get_pid()
        print(f"process   : running (pid {pid})")
    else:
        print("process   : stopped")

    if health_ok(cfg.llama_url) and models_ok(cfg.llama_url):
        print(f"health    : OK ({cfg.llama_url}/health)")
    else:
        print(f"health    : DOWN/UNHEALTHY")

    # Lock status
    lockfile = cfg.state_dir / "gpu.lock"
    try:
        lock_fd = os.open(str(lockfile), os.O_CREAT | os.O_WRONLY)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
        print("gpu lock  : free")
    except OSError:
        print("gpu lock  : HELD (a gpurun job is active)")

    # Recent events
    events_file = cfg.state_dir / "events.jsonl"
    if events_file.exists() and events_file.stat().st_size > 0:
        print("\nlast events:")
        with open(events_file) as f:
            lines = f.readlines()
        for line in lines[-8:]:
            try:
                ev = json.loads(line)
                keys = {k: v for k, v in ev.items() if k not in ("ts", "event", "run")}
                print(f"  {ev['ts']}  {ev['event']}  {keys}")
            except (json.JSONDecodeError, KeyError):
                pass

    return 0


# -------------------------------------------------------- command: log
def cmd_log(cfg: Config, args: argparse.Namespace) -> int:
    """Show or tail the human-readable log."""
    logger = EventLogger(cfg.state_dir)
    if args.follow:
        lines = logger.tail(n=getattr(args, "lines", 40))
        for line in lines:
            print(line)
        logger.follow()
    else:
        lines = logger.tail(n=getattr(args, "lines", 40))
        for line in lines:
            print(line)
    return 0


# ------------------------------------------------------- command: on/off
def cmd_on(cfg: Config, args: argparse.Namespace) -> int:
    """Ensure llama-server is running and healthy."""
    desired_file = cfg.state_dir / "desired"
    desired_file.write_text("on")

    driver = ProcessDriver(
        start_cmd=cfg.llama_start_cmd or "",
        proc_pattern=cfg.llama_proc_pattern,
        health_url=cfg.llama_url,
        port=cfg.llama_port,
        state_dir=cfg.state_dir,
        stop_timeout=cfg.stop_timeout,
    )

    logger = EventLogger(cfg.state_dir, f"manual-{os.getpid()}")

    if driver.is_running() and health_ok(cfg.llama_url):
        print("[gpurun] llama-server already running and healthy", file=sys.stderr)
        return 0

    if driver.is_running():
        print("[gpurun] llama-server unhealthy; restarting", file=sys.stderr)
        logger.evt("llama_stopping")
        if not driver.stop():
            print("[gpurun] llama-server did not terminate", file=sys.stderr)
            return 95
        logger.evt("llama_stopped")

    success, code = driver.restore(cfg.health_timeout)
    if not success:
        return code

    print(f"[gpurun] llama-server up and healthy ({cfg.llama_url})", file=sys.stderr)
    return 0


def cmd_off(cfg: Config, args: argparse.Namespace) -> int:
    """Stop llama-server and disable auto-restart."""
    desired_file = cfg.state_dir / "desired"
    desired_file.write_text("off")

    driver = ProcessDriver(
        start_cmd=cfg.llama_start_cmd or "",
        proc_pattern=cfg.llama_proc_pattern,
        health_url=cfg.llama_url,
        port=cfg.llama_port,
        state_dir=cfg.state_dir,
        stop_timeout=cfg.stop_timeout,
    )

    logger = EventLogger(cfg.state_dir, f"manual-{os.getpid()}")

    if not driver.is_running():
        print("[gpurun] llama-server already stopped (guard disabled until 'gpurun on')", file=sys.stderr)
        return 0

    logger.evt("llama_stopping")
    if not driver.stop():
        print("[gpurun] llama-server did not terminate", file=sys.stderr)
        return 95
    logger.evt("llama_stopped")
    print("[gpurun] llama-server stopped; guard disabled until 'gpurun on'", file=sys.stderr)
    return 0


# ------------------------------------------------------ command: use
def cmd_use(cfg: Config, args: argparse.Namespace) -> int:
    """Switch the model script."""
    target = args.target
    if not target:
        print("[gpurun] usage: gpurun use NAME|PATH", file=sys.stderr)
        return 99

    # Resolve target
    model_link = Path.home() / ".config" / "gpu-orchestrator" / "current-model.sh"

    if Path(target).is_file():
        resolved = Path(target).resolve()
    else:
        resolved = None
        for d in cfg.model_dirs:
            for suffix in ["", ".sh"]:
                candidate = d / (target + suffix)
                if candidate.exists():
                    resolved = candidate.resolve()
                    break
            if resolved:
                break

    if not resolved or not os.access(str(resolved), os.X_OK):
        print(f"[gpurun] model script not found or not executable: {target}", file=sys.stderr)
        return 99

    model_link.parent.mkdir(parents=True, exist_ok=True)
    if model_link.exists() or model_link.is_symlink():
        model_link.unlink()
    model_link.symlink_to(resolved)

    cfg.save({"llama": {"start_cmd": str(resolved)}})

    logger = EventLogger(cfg.state_dir, f"use-{os.getpid()}")
    logger.evt("model_switched", target=str(resolved))

    # Restart if needed
    desired_file = cfg.state_dir / "desired"
    desired = "on"
    if desired_file.exists():
        desired = desired_file.read_text().strip()

    driver = ProcessDriver(
        start_cmd=str(resolved),
        proc_pattern=cfg.llama_proc_pattern,
        health_url=cfg.llama_url,
        port=cfg.llama_port,
        state_dir=cfg.state_dir,
        stop_timeout=cfg.stop_timeout,
    )

    if driver.is_running():
        logger.evt("llama_stopping")
        if not driver.stop():
            return 95
        logger.evt("llama_stopped")

    if desired == "on":
        success, code = driver.restore(cfg.health_timeout)
        if not success:
            return code

    print(f"[gpurun] now serving: {resolved}", file=sys.stderr)
    return 0


# ----------------------------------------------------- command: detect
def cmd_detect(cfg: Config, args: argparse.Namespace) -> int:
    """Show detected environment without modifying anything."""
    from .config import _find_llama_binary, _find_running_llama

    print("=== GPU Orchestrator Detection ===\n")

    binary = _find_llama_binary()
    print(f"llama-server binary: {'FOUND: ' + str(binary) if binary else 'NOT FOUND'}")

    url = cfg.llama_url
    try:
        import urllib.request

        req = urllib.request.Request(f"{url}/health")
        with urllib.request.urlopen(req, timeout=3) as resp:
            status = resp.status
        print(f"HTTP endpoint       : {url} -> {status}")

        req2 = urllib.request.Request(f"{url}/v1/models")
        req2.add_header("Accept", "application/json")
        with urllib.request.urlopen(req2, timeout=3) as resp:
            data = json.loads(resp.read())
            models = data.get("data", [])
            if models:
                m = models[0]
                print(f"  model             : {m.get('id', m.get('name', '?'))}")
                meta = m.get("meta", {})
                size = meta.get("size", 0)
                if isinstance(size, (int, float)) and size > 0:
                    print(f"  size              : {size / (1024**3):.1f} GB")
    except Exception as e:
        print(f"HTTP endpoint       : {url} -> NOT REACHABLE ({e})")

    info = _find_running_llama(cfg.llama_url, cfg.llama_port)
    if info:
        print(f"Running process     : YES")
        print(f"  proc_pattern      : {info['proc_pattern']}")
        print(f"  start_cmd         : {info['start_cmd'][:200]}")
    else:
        print("Running process     : NO (server not responding)")

    # GPU
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            print("\nGPUs:")
            for line in result.stdout.strip().split("\n"):
                print(f"  {line.strip()}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("\nGPUs: nvidia-smi not available")

    # Claude hook
    for d in [Path.home() / ".claude"]:
        settings = d / "settings.json"
        if settings.exists():
            try:
                with open(settings) as f:
                    s = json.load(f)
                hooks = s.get("hooks", {}).get("PreToolUse", [])
                found = any(
                    "gpu-intercept" in h.get("hooks", [{}])[0].get("command", "")
                    for h in hooks
                )
                tag = "FOUND" if found else "NOT CONFIGURED"
                print(f"Claude hook         : {tag} in {settings}")
            except (json.JSONDecodeError, KeyError, IndexError):
                pass

    # Cron guard
    try:
        result = subprocess.run(
            ["crontab", "-l"], capture_output=True, text=True, timeout=5,
        )
        tag = "INSTALLED" if "gpu-orchestrator" in result.stdout else "NOT INSTALLED"
        print(f"\nCron guard          : {tag}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("\nCron guard          : crontab not available")

    print(f"\nGPU patterns ({len(cfg.all_patterns)} total):")
    for p in cfg.all_patterns:
        if p:
            print(f"  - {p}")

    return 0


# ---------------------------------------------------- command: setup
def cmd_setup(cfg: Config, args: argparse.Namespace) -> int:
    """Interactive setup wizard."""
    from .config import _find_llama_binary, _find_running_llama

    print("=== GPU Orchestrator Setup ===\n")

    # Step 1: Detect
    print("Step 1: Detecting environment...\n")
    binary = _find_llama_binary()
    print(f"  llama-server binary: {binary or 'NOT FOUND'}")

    info = _find_running_llama(cfg.llama_url, cfg.llama_port)
    if info:
        print(f"  Running server     : YES")
        print(f"  proc_pattern       : {info['proc_pattern']}")
        print(f"  start_cmd          : {info['start_cmd'][:200]}")
    else:
        print("  Running server     : NO")

    # Step 2: Configure
    print("\nStep 2: Configuration\n")

    url = input(f"  HTTP URL [{cfg.llama_url}]: ").strip()
    if url:
        cfg.save({"llama": {"url": url}})
        cfg._raw["llama"]["url"] = url
        try:
            port = int(url.split(":")[-1].split("/")[0])
            cfg.save({"llama": {"port": port}})
            cfg._raw["llama"]["port"] = port
        except ValueError:
            pass

    if info and info["proc_pattern"]:
        pattern = input(f"  Process pattern [{info['proc_pattern']}]: ").strip()
        if pattern:
            cfg.save({"llama": {"proc_pattern": pattern}})
            cfg._raw["llama"]["proc_pattern"] = pattern

    if info and info["start_cmd"]:
        prompt = f"  Start command [{info['start_cmd'][:100]}...]: "
        start_cmd = input(prompt).strip()
        if start_cmd:
            cfg.save({"llama": {"start_cmd": start_cmd}})
            cfg._raw["llama"]["start_cmd"] = start_cmd
    elif binary:
        print(f"\n  No running server found. Binary at: {binary}")
        start_cmd = input(f"  Start command (e.g. '{binary} --help'): ").strip()
        if start_cmd:
            cfg.save({"llama": {"start_cmd": start_cmd}})
            cfg._raw["llama"]["start_cmd"] = start_cmd
        else:
            print("  WARNING: No start command. Set with: gpurun use <script>")

    # Step 3: Save
    print("\nStep 3: Saving configuration")
    print(f"  Config: {Path.home() / '.config' / 'gpu-orchestrator' / 'config.yaml'}")

    # Step 4: Claude hook
    print("\nStep 4: Claude Code hook")
    if input("  Install PreToolUse hook? [Y/n]: ").strip().lower() != "n":
        install_claude_hook(cfg)

    # Step 5: Cron
    print("\nStep 5: Cron guard")
    if input("  Install cron self-heal? [Y/n]: ").strip().lower() != "n":
        install_cron_guard()

    print("\nSetup complete!")
    print("  Try: gpurun status")
    print("  Try: gpurun on")
    print("  Try: gpurun --auto -- echo hello")
    return 0


def install_claude_hook(cfg: Config) -> None:
    """Install PreToolUse hook in Claude Code settings."""
    hook_script = (
        Path.home() / ".local" / "share" / "gpu-orchestrator"
        / "claude" / "gpu-intercept.sh"
    )
    if not hook_script.exists():
        print(f"  WARNING: hook script missing at {hook_script}")
        return

    for d in [Path.home() / ".claude"]:
        settings = d / "settings.json"
        if not settings.exists():
            continue

        try:
            with open(settings) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {}

        backup = settings.with_suffix(f".bak-{time.strftime('%Y%m%dT%H%M%S')}")
        try:
            settings.rename(backup)
        except OSError:
            pass
        data = {}
        try:
            with open(backup) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

        hooks = data.get("hooks", {}).get("PreToolUse", [])
        already = any(
            "gpu-intercept" in h.get("hooks", [{}])[0].get("command", "")
            for h in hooks
        )

        if not already:
            data.setdefault("hooks", {}).setdefault("PreToolUse", []).append({
                "matcher": "Bash",
                "hooks": [{"type": "command", "command": str(hook_script), "timeout": 15}],
            })
            with open(settings, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  Hook installed in {settings}")
        else:
            print(f"  Hook already present in {settings}")


def install_cron_guard() -> None:
    """Install cron guard entries."""
    gpurun_bin = Path.home() / ".local" / "bin" / "gpurun"
    if not gpurun_bin.exists():
        print(f"  WARNING: gpurun not found at {gpurun_bin}")
        return

    marker_s = "# >>> gpu-orchestrator (managed) >>>"
    marker_e = "# <<< gpu-orchestrator <<<"
    g = str(gpurun_bin)

    try:
        result = subprocess.run(["crontab", "-l"], capture_output=True, text=True, timeout=5)
        existing = result.stdout if result.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        existing = ""

    lines = existing.split("\n")
    filtered = []
    skip = False
    for line in lines:
        if marker_s in line:
            skip = True
            continue
        if marker_e in line:
            skip = False
            continue
        if skip:
            continue
        filtered.append(line)

    filtered.append(marker_s)
    filtered.append(f"@reboot {g} guard")
    filtered.append(f"*/2 * * * * {g} guard")
    filtered.append(marker_e)

    new_cron = "\n".join(filtered).strip() + "\n"
    proc = subprocess.Popen(["crontab", "-"], stdin=subprocess.PIPE, text=True)
    proc.communicate(input=new_cron)
    print("  Cron guard installed (@reboot + every 2 min)")


# ------------------------------------------------- command: __match
def cmd_match(cfg: Config, args: argparse.Namespace) -> int:
    """Test if a command matches GPU patterns (hook API).

    Prints the matched pattern (exit 0) or nothing (exit 1). With
    ``--with-decision`` the configured hook decision is printed first, on
    its own line, then the pattern — so the hook can read both from a single
    invocation without parsing config.yaml in shell.
    """
    pattern = matches_gpu(" ".join(args.cmd), cfg.extra_patterns)
    if not pattern:
        return 1
    if getattr(args, "with_decision", False):
        print(cfg.hook_decision)
    print(pattern)
    return 0


# ------------------------------------------------- command: off-intercept
def cmd_off_intercept(cfg: Config, args: argparse.Namespace) -> int:
    """Disable the PreToolUse gpu-intercept hook; llama-server keeps running."""
    hook_script = (
        Path.home() / ".local" / "share" / "gpu-orchestrator"
        / "claude" / "gpu-intercept.sh"
    )
    if not hook_script.exists():
        print(f"[gpurun] hook script missing at {hook_script}", file=sys.stderr)
        return 1

    for d in [Path.home() / ".claude"]:
        settings = d / "settings.json"
        if not settings.exists():
            continue

        try:
            with open(settings) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            print(f"[gpurun] read failed: {settings}", file=sys.stderr)
            continue

        hooks = data.get("hooks", {}).get("PreToolUse", [])
        before = len(hooks)
        data["hooks"]["PreToolUse"] = [
            h for h in hooks
            if "gpu-intercept" not in h.get("hooks", [{}])[0].get("command", "")
        ]
        after = len(data["hooks"]["PreToolUse"])

        if before != after:
            with open(settings, "w") as f:
                json.dump(data, f, indent=2)
            print(f"[gpurun] removed gpu-intercept hook from {settings}")
        else:
            print(f"[gpurun] hook already absent in {settings}")

    print("[gpurun] intercept disabled — llama-server stays running")
    return 0


# ------------------------------------------------- command: on-intercept
def cmd_on_intercept(cfg: Config, args: argparse.Namespace) -> int:
    """Re-enable the PreToolUse gpu-intercept hook."""
    hook_script = (
        Path.home() / ".local" / "share" / "gpu-orchestrator"
        / "claude" / "gpu-intercept.sh"
    )
    if not hook_script.exists():
        print(f"[gpurun] hook script missing at {hook_script}", file=sys.stderr)
        return 1

    for d in [Path.home() / ".claude"]:
        settings = d / "settings.json"
        if not settings.exists():
            continue

        try:
            with open(settings) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            print(f"[gpurun] read failed: {settings}", file=sys.stderr)
            continue

        # Preserve existing backup so we can re-read original if needed
        backup = settings.with_suffix(".bak-" + time.strftime("%Y%m%dT%H%M%S"))
        try:
            settings.rename(backup)
        except OSError:
            pass
        data = {}
        try:
            with open(backup) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

        hooks = data.get("hooks", {}).get("PreToolUse", [])
        already = any(
            "gpu-intercept" in h.get("hooks", [{}])[0].get("command", "")
            for h in hooks
        )

        if not already:
            data.setdefault("hooks", {}).setdefault("PreToolUse", []).append({
                "matcher": "Bash",
                "hooks": [{"type": "command", "command": str(hook_script), "timeout": 15}],
            })
            with open(settings, "w") as f:
                json.dump(data, f, indent=2)
            print(f"[gpurun] re-added gpu-intercept hook to {settings}")
        else:
            print(f"[gpurun] hook already present in {settings}")

    print("[gpurun] intercept re-enabled — future GPU commands will be wrapped")
    return 0


# -------------------------------------------------------- CLI entry point
def main() -> None:
    """Main CLI entry point."""
    # Ignore SIGHUP so gpurun survives tmux death and can clean up children.
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    parser = argparse.ArgumentParser(
        prog="gpurun",
        description="GPU job supervisor — pauses llama.cpp server around VRAM-heavy jobs",
        add_help=True,
    )
    subparsers = parser.add_subparsers(dest="command")

    # run
    rp = subparsers.add_parser("run", help="Run command with GPU management")
    rp.add_argument("--auto", action="store_true")
    rp.add_argument("--no-restart", action="store_true")
    rp.add_argument("-c", dest="command_string")
    rp.add_argument("argv", nargs="*")
    rp.add_argument("--timeout", type=int, default=None, help="Max seconds for the command (default: config value)")
    # argparse handles -- automatically, but we need to strip it from argv

    subparsers.add_parser("status", help="Show current state")
    subparsers.add_parser("on", help="Start/restore llama-server")
    subparsers.add_parser("off", help="Stop llama-server")

    lp = subparsers.add_parser("log", help="View logs")
    lp.add_argument("-f", "--follow", action="store_true")
    lp.add_argument("lines", nargs="?", type=int, default=40)

    up = subparsers.add_parser("use", help="Switch model script")
    up.add_argument("target")

    subparsers.add_parser("guard", help="Cron self-heal")
    subparsers.add_parser("detect", help="Show detected environment")
    subparsers.add_parser("setup", help="Interactive setup wizard")

    oi = subparsers.add_parser("off-intercept", help="Disable Claude hook; llama-server stays running")
    ni = subparsers.add_parser("on-intercept", help="Re-enable Claude hook for GPU command wrapping")

    mp = subparsers.add_parser("__match", help="Pattern test (hook API)")
    mp.add_argument("--with-decision", action="store_true",
                    help="Print configured hook decision on first line, pattern on second")
    mp.add_argument("cmd", nargs="+")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    cfg = load()

    cmd_map = {
        "run": cmd_run,
        "status": cmd_status,
        "log": cmd_log,
        "on": cmd_on,
        "off": cmd_off,
        "use": cmd_use,
        "guard": cmd_guard,
        "detect": cmd_detect,
        "setup": cmd_setup,
        "off-intercept": cmd_off_intercept,
        "on-intercept": cmd_on_intercept,
        "__match": cmd_match,
    }

    handler = cmd_map.get(args.command)
    if handler:
        sys.exit(handler(cfg, args))
    else:
        parser.print_help()
        sys.exit(99)


if __name__ == "__main__":
    main()
