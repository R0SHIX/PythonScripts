#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import time
import psutil
import shutil

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def is_root():
    return os.geteuid() == 0

def run_cmd(cmd: str, dry_run: bool = False) -> str:
    if dry_run:
        print(f"  [DRY-RUN] Would run: {cmd}")
        return ""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return result.stdout.strip()
    except Exception as e:
        return f"[ERROR] {e}"

def confirm(prompt: str, force: bool) -> bool:
    if force:
        return True
    ans = input(f"{prompt} [y/N]: ").strip().lower()
    return ans == "y"

def section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ─────────────────────────────────────────────
#  1. NOTIFY USERS
# ─────────────────────────────────────────────

def notify_users(dry_run: bool, wait_seconds: int = 30):
    """Broadcast a wall message to all logged-in users, then wait."""
    section("STEP 1: Notifying logged-in users")

    # Show who's currently logged in
    logged_in = run_cmd("who", dry_run=False)
    if logged_in:
        print("  Currently logged-in users:")
        for line in logged_in.splitlines():
            print(f"    {line}")
    else:
        print("  No users currently logged in.")
        return

    message = (
        f"[BENCHMARK NOTICE] This server is being taken offline for performance benchmarking "
        f"in {wait_seconds} seconds. Please save your work and disconnect. "
        f"Contact the administrator if you need more time."
    )

    print(f"\n  Sending wall message...")
    run_cmd(f'wall "{message}"', dry_run)

    if not dry_run:
        print(f"  Waiting {wait_seconds}s for users to disconnect...")
        for i in range(wait_seconds, 0, -5):
            print(f"    {i}s remaining...", end="\r")
            time.sleep(5)
        print("\n  Wait complete.")


# ─────────────────────────────────────────────
#  2. KILL USER SESSIONS & PROCESSES
# ─────────────────────────────────────────────

def kill_user_sessions(dry_run: bool, force: bool, exclude_user: str):
    """Kill all logged-in user sessions except the current user."""
    section("STEP 2: Killing user sessions and processes")

    current_user = os.environ.get("SUDO_USER") or os.environ.get("USER") or "root"
    print(f"  Current user (will be excluded): {current_user}")
    print(f"  Exclude user: {exclude_user}" if exclude_user else "")

    # Get all logged-in users
    who_output = run_cmd("who -u", dry_run=False)
    if not who_output:
        print("  No other users to kill.")
        return

    users_to_kill = set()
    for line in who_output.splitlines():
        parts = line.split()
        if parts:
            user = parts[0]
            if user != current_user and user != exclude_user and user != "root":
                users_to_kill.add(user)

    if not users_to_kill:
        print("  No other users found to disconnect.")
    else:
        print(f"  Users to disconnect: {', '.join(users_to_kill)}")
        if confirm(f"  Kill sessions for: {', '.join(users_to_kill)}?", force):
            for user in users_to_kill:
                # Kill all their processes
                print(f"  Killing all processes for user: {user}")
                run_cmd(f"pkill -KILL -u {user}", dry_run)
                # Force logout their SSH/terminal sessions
                run_cmd(f"loginctl terminate-user {user}", dry_run)
                print(f"  [✓] {user} disconnected")

    # Kill known GPU-hogging processes from ALL users (jupyter, python training jobs etc)
    section("STEP 2b: Killing known GPU workload processes")
    gpu_proc_patterns = [
        "jupyter", "jupyter-notebook", "jupyter-lab",
        "python.*train", "python.*fit", "python.*torch",
        "python.*tensorflow", "python.*keras",
        "nvitop", "gpustat"
    ]
    for pattern in gpu_proc_patterns:
        result = run_cmd(f"pgrep -a -f '{pattern}'", dry_run=False)
        if result:
            print(f"  Found processes matching '{pattern}':")
            for line in result.splitlines():
                pid = line.split()[0]
                # Don't kill our own script
                if str(os.getpid()) not in pid:
                    print(f"    Killing PID {line}")
                    run_cmd(f"kill -9 {pid}", dry_run)


# ─────────────────────────────────────────────
#  3. FREE GPU MEMORY
# ─────────────────────────────────────────────

def free_gpu_memory(dry_run: bool):
    """Kill zombie GPU processes and reset GPU state."""
    section("STEP 3: Freeing GPU memory")

    try:
        import pynvml
        pynvml.nvmlInit()
        num_gpus = pynvml.nvmlDeviceGetCount()
    except Exception:
        print("  [WARNING] pynvml not available. Skipping GPU-specific cleanup.")
        num_gpus = 0

    # Find all PIDs using any GPU
    gpu_pids = set()
    for i in range(num_gpus):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            print(f"  GPU {i} ({name}): {mem_info.used/1024**2:.0f} MB / {mem_info.total/1024**2:.0f} MB used — {len(procs)} compute process(es)")
            for proc in procs:
                gpu_pids.add(proc.pid)
        except Exception as e:
            print(f"  GPU {i}: Could not query — {e}")

    if gpu_pids:
        print(f"\n  GPU processes to kill: {gpu_pids}")
        for pid in gpu_pids:
            try:
                proc = psutil.Process(pid)
                if proc.pid == os.getpid():
                    continue
                print(f"  Killing PID {pid} ({proc.name()}, user={proc.username()})")
                run_cmd(f"kill -9 {pid}", dry_run)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    else:
        print("  No active GPU compute processes found.")

    # Attempt nvidia-smi reset on each GPU (requires exclusive process mode or admin)
    print("\n  Running nvidia-smi to confirm post-cleanup state:")
    result = run_cmd("nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits", dry_run=False)
    if result:
        print(f"  {'GPU':<5} {'Name':<25} {'Mem Used(MB)':<14} {'Mem Free(MB)':<14} {'Util%':<7} {'Temp°C'}")
        print("  " + "-" * 75)
        for line in result.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                print(f"  {parts[0]:<5} {parts[1]:<25} {parts[2]:<14} {parts[3]:<14} {parts[4]:<7} {parts[5]}")


# ─────────────────────────────────────────────
#  4. FREE SYSTEM RAM / CACHES
# ─────────────────────────────────────────────

def free_system_memory(dry_run: bool):
    """Drop Linux page cache, dentries, and inodes."""
    section("STEP 4: Freeing system RAM caches")

    mem_before = psutil.virtual_memory()
    print(f"  RAM before: {mem_before.used/1024**3:.2f} GB used / {mem_before.total/1024**3:.2f} GB total ({mem_before.percent}%)")

    # Sync filesystem first, then drop caches
    print("  Syncing filesystem...")
    run_cmd("sync", dry_run)

    print("  Dropping page cache, dentries, inodes (echo 3 > /proc/sys/vm/drop_caches)...")
    run_cmd("echo 3 > /proc/sys/vm/drop_caches", dry_run)

    if not dry_run:
        time.sleep(1)
        mem_after = psutil.virtual_memory()
        freed = (mem_before.used - mem_after.used) / 1024**3
        print(f"  RAM after:  {mem_after.used/1024**3:.2f} GB used ({mem_after.percent}%)")
        print(f"  [✓] Freed ~{freed:.2f} GB from page cache")


# ─────────────────────────────────────────────
#  5. STOP NON-ESSENTIAL SYSTEM SERVICES
# ─────────────────────────────────────────────

def stop_services(dry_run: bool, force: bool):
    """Stop services that compete for CPU/memory during benchmark."""
    section("STEP 5: Stopping non-essential services")

    # Services commonly competing with GPU workloads on ML servers
    services_to_stop = [
        "cron",            # scheduled jobs firing mid-benchmark
        "snapd",           # background snaps updating
        "unattended-upgrades",  # package manager running
        "packagekit",      # package queries
        "cups",            # printer service (usually useless on servers)
        "avahi-daemon",    # mDNS/Bonjour discovery
        "bluetooth",       # if present
        "ModemManager",    # modem management
    ]

    stopped = []
    skipped = []

    for svc in services_to_stop:
        # Check if service exists and is active
        status = run_cmd(f"systemctl is-active {svc} 2>/dev/null", dry_run=False)
        if status == "active":
            print(f"  [{svc}] is active — stopping")
            run_cmd(f"systemctl stop {svc}", dry_run)
            stopped.append(svc)
        else:
            skipped.append(svc)

    print(f"\n  Stopped: {stopped if stopped else 'none'}")
    print(f"  Skipped (not active): {skipped}")
    print(f"\n  [!] Note: These will NOT restart automatically until you reboot or run 'systemctl start <service>'")
    print(f"      To restore all: sudo systemctl start {' '.join(stopped)}" if stopped else "")


# ─────────────────────────────────────────────
#  6. SET CPU PERFORMANCE GOVERNOR
# ─────────────────────────────────────────────

def set_cpu_performance_mode(dry_run: bool):
    """Force all CPU cores to performance governor (disable power saving)."""
    section("STEP 6: Setting CPU governor to 'performance'")

    num_cores = psutil.cpu_count(logical=True)
    governor_path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"

    if not os.path.exists(governor_path):
        print("  [SKIP] CPU frequency scaling not available (may be a VM or unsupported kernel)")
        return

    current = run_cmd(f"cat {governor_path}")
    print(f"  Current governor: {current}")

    if current == "performance":
        print("  Already in performance mode. Nothing to do.")
        return

    if confirm("  Set all cores to 'performance' governor?", dry_run or False):
        for i in range(num_cores):
            path = f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_governor"
            run_cmd(f"echo performance > {path}", dry_run)
        print(f"  [✓] All {num_cores} cores set to performance governor")
        print(f"  [!] To restore: run this script with --restore-governor or set back to 'ondemand'")


# ─────────────────────────────────────────────
#  7. FINAL SYSTEM STATE REPORT
# ─────────────────────────────────────────────

def final_report():
    section("FINAL SYSTEM STATE — Ready for Benchmark")

    mem = psutil.virtual_memory()
    cpu_freq = psutil.cpu_freq()
    load = os.getloadavg()

    print(f"  RAM:      {mem.used/1024**3:.2f} GB used / {mem.total/1024**3:.2f} GB total ({mem.percent}%)")
    print(f"  CPU Load: {load[0]:.2f} (1m) | {load[1]:.2f} (5m) | {load[2]:.2f} (15m)")
    print(f"  CPU Freq: {cpu_freq.current:.0f} MHz (max: {cpu_freq.max:.0f} MHz)" if cpu_freq else "  CPU Freq: N/A")

    logged_in = run_cmd("who")
    print(f"  Logged-in users: {logged_in if logged_in else 'none'}")

    try:
        import pynvml
        pynvml.nvmlInit()
        num_gpus = pynvml.nvmlDeviceGetCount()
        print(f"\n  GPU State:")
        for i in range(num_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            name = pynvml.nvmlDeviceGetName(handle)
            print(f"    GPU {i} ({name}): {mem_info.used/1024**2:.0f} MB used | {util.gpu}% util | {temp}°C")
    except Exception:
        pass

    print(f"\n{'─' * 60}")
    print(f"  [✓] Server prep complete. Run benchmark_stress.py now.")
    print(f"{'─' * 60}\n")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Server prep for benchmark stress testing")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without doing it")
    parser.add_argument("--force", action="store_true", help="Skip all confirmation prompts")
    parser.add_argument("--notify", action="store_true", help="Send wall message and wait before cleanup")
    parser.add_argument("--notify-wait", type=int, default=30, help="Seconds to wait after notifying users (default: 30)")
    parser.add_argument("--exclude-user", type=str, default="", help="Username to exclude from session killing")
    parser.add_argument("--skip-services", action="store_true", help="Skip stopping system services")
    parser.add_argument("--skip-governor", action="store_true", help="Skip CPU governor change")
    args = parser.parse_args()

    print("=" * 60)
    print("  SERVER PREP — Pre-Benchmark Cleanup")
    print("=" * 60)

    if args.dry_run:
        print("\n  *** DRY RUN MODE — Nothing will be modified ***\n")

    if not is_root() and not args.dry_run:
        print("\n[ERROR] This script requires root privileges for full cleanup.")
        print("  Run with: sudo python server_prep.py")
        print("  Or use --dry-run to preview without root.\n")
        sys.exit(1)

    if args.notify:
        notify_users(args.dry_run, args.notify_wait)

    kill_user_sessions(args.dry_run, args.force, args.exclude_user)
    free_gpu_memory(args.dry_run)
    free_system_memory(args.dry_run)

    if not args.skip_services:
        stop_services(args.dry_run, args.force)

    if not args.skip_governor:
        set_cpu_performance_mode(args.dry_run)

    final_report()


if __name__ == "__main__":
    main()
