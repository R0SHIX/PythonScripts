#!/usr/bin/env python3

import argparse
import multiprocessing
import os
import sys
import threading
import time
from datetime import datetime

import psutil

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch not found. GPU stress will be disabled.")

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False


# ─────────────────────────────────────────────
#  GPU STRESS WORKER
# ─────────────────────────────────────────────

class HeavyModel(nn.Module):
    """Simulates a realistic deep learning model with Conv + Linear layers."""
    def __init__(self, input_size=224, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def gpu_stress_worker(gpu_id: int, stats: dict, stop_event: threading.Event,
                      batch_size: int, input_size: int):
    """
    Runs forward + backward passes continuously on a single GPU.
    Designed to maximize VRAM usage and compute throughput.
    """
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    model = HeavyModel(input_size=input_size).to(device)
    model.train()

    # Use mixed precision for realistic workload + higher VRAM pressure
    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    iterations = 0
    total_samples = 0
    start_time = time.time()

    while not stop_event.is_set():
        try:
            # Generate random input (simulates image batch)
            inputs = torch.randn(batch_size, 3, input_size, input_size, device=device)
            targets = torch.randint(0, 1000, (batch_size,), device=device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            iterations += 1
            total_samples += batch_size
            elapsed = time.time() - start_time

            stats[gpu_id] = {
                "iterations": iterations,
                "throughput": total_samples / elapsed,
                "loss": loss.item(),
                "vram_used_mb": torch.cuda.memory_allocated(device) / 1024**2,
                "vram_total_mb": torch.cuda.get_device_properties(device).total_memory / 1024**2,
            }

        except torch.cuda.OutOfMemoryError:
            print(f"\n[GPU {gpu_id}] OOM — reducing batch size to {batch_size // 2}")
            batch_size = batch_size // 2
            torch.cuda.empty_cache()
            if batch_size < 1:
                print(f"[GPU {gpu_id}] Batch size too small. Exiting GPU worker.")
                break
        except Exception as e:
            print(f"\n[GPU {gpu_id}] Error: {e}")
            break

    torch.cuda.empty_cache()


# ─────────────────────────────────────────────
#  CPU STRESS WORKER
# ─────────────────────────────────────────────

def cpu_stress_worker(worker_id: int, stats: dict, stop_event: multiprocessing.Event):
    """
    Saturates a single CPU core with floating-point matrix operations.
    Pure Python + no dependencies for portability.
    """
    import math
    iterations = 0
    start = time.time()

    # Use large arrays of floats — cache-busting
    size = 512
    a = [float(i % 100) for i in range(size * size)]
    b = [float((i + 1) % 100) for i in range(size * size)]

    while not stop_event.is_set():
        # Matrix multiply simulation (dot products)
        result = 0.0
        for i in range(0, len(a), 64):
            chunk = a[i:i+64]
            bchunk = b[i:i+64]
            result += sum(x * y + math.sqrt(abs(x - y) + 1e-9) for x, y in zip(chunk, bchunk))

        iterations += 1
        elapsed = time.time() - start
        stats[worker_id] = {
            "iterations": iterations,
            "ops_per_sec": iterations / elapsed,
        }


# ─────────────────────────────────────────────
#  MONITORING & REPORTING
# ─────────────────────────────────────────────

def get_gpu_stats_nvml():
    """Fetch GPU utilization and temperature via NVML."""
    if not NVML_AVAILABLE:
        return {}
    results = {}
    count = pynvml.nvmlDeviceGetCount()
    for i in range(count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        results[i] = {
            "gpu_util": util.gpu,
            "mem_util": util.memory,
            "temp_c": temp,
            "mem_used_mb": mem_info.used / 1024**2,
            "mem_total_mb": mem_info.total / 1024**2,
        }
    return results


def monitor_and_report(gpu_stats: dict, cpu_stats: dict, stop_event: threading.Event,
                       num_gpus: int, num_cpu_workers: int, duration: float, log_file: str):
    """Prints a live dashboard and writes results to a log file."""
    start_time = time.time()
    log_lines = []

    while not stop_event.is_set():
        elapsed = time.time() - start_time
        if duration > 0 and elapsed >= duration:
            stop_event.set()
            break

        os.system("clear" if os.name == "posix" else "cls")

        now = datetime.now().strftime("%H:%M:%S")
        remaining = f"{max(0, duration - elapsed):.0f}s remaining" if duration > 0 else "∞ (Ctrl+C to stop)"
        
        lines = []
        lines.append("=" * 70)
        lines.append(f"  BENCHMARK STRESS TEST  |  {now}  |  {elapsed:.0f}s elapsed  |  {remaining}")
        lines.append("=" * 70)

        # GPU section
        if num_gpus > 0 and TORCH_AVAILABLE:
            nvml_data = get_gpu_stats_nvml()
            lines.append(f"\n  {'GPU':<6} {'Util%':<8} {'Temp°C':<9} {'VRAM Used':<14} {'VRAM Total':<14} {'Throughput (img/s)'}")
            lines.append("  " + "-" * 66)
            for gid in range(num_gpus):
                torch_s = gpu_stats.get(gid, {})
                nvml_s = nvml_data.get(gid, {})
                vram_used = torch_s.get("vram_used_mb", nvml_s.get("mem_used_mb", 0))
                vram_total = torch_s.get("vram_total_mb", nvml_s.get("mem_total_mb", 0))
                util = nvml_s.get("gpu_util", "N/A")
                temp = nvml_s.get("temp_c", "N/A")
                throughput = torch_s.get("throughput", 0)
                lines.append(f"  {gid:<6} {str(util)+'%':<8} {str(temp)+'°C':<9} {vram_used:.0f} MB{'':<6} {vram_total:.0f} MB{'':<6} {throughput:.1f}")

        # CPU section
        if num_cpu_workers > 0:
            cpu_pct = psutil.cpu_percent(percpu=False)
            mem = psutil.virtual_memory()
            cpu_freq = psutil.cpu_freq()
            freq_str = f"{cpu_freq.current:.0f} MHz" if cpu_freq else "N/A"
            
            lines.append(f"\n  CPU Overall: {cpu_pct}%  |  Frequency: {freq_str}  |  RAM: {mem.used/1024**3:.1f}/{mem.total/1024**3:.1f} GB ({mem.percent}%)")
            lines.append(f"  CPU Workers Active: {num_cpu_workers} / {psutil.cpu_count(logical=True)} logical cores")

        lines.append("\n  [Ctrl+C] to stop and save report")
        lines.append("=" * 70)

        print("\n".join(lines))
        log_lines.append({"time": elapsed, "gpu": dict(gpu_stats), "cpu_pct": psutil.cpu_percent()})

        time.sleep(2)

    # Write log
    with open(log_file, "w") as f:
        f.write(f"Benchmark Run: {datetime.now().isoformat()}\n")
        f.write(f"Duration: {time.time() - start_time:.1f}s\n")
        f.write(f"GPUs: {num_gpus} | CPU Workers: {num_cpu_workers}\n\n")
        for entry in log_lines:
            f.write(str(entry) + "\n")
    print(f"\n[✓] Report saved to: {log_file}")


# ─────────────────────────────────────────────
#  MAIN ENTRY
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU + CPU Stress Benchmark")
    parser.add_argument("--duration", type=float, default=0,
                        help="Duration in seconds (0 = run until Ctrl+C)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size per GPU (default: 32, increase for higher VRAM usage)")
    parser.add_argument("--input-size", type=int, default=224,
                        help="Input image size (default: 224, use 512 for heavier load)")
    parser.add_argument("--gpu-only", action="store_true", help="Only stress GPUs")
    parser.add_argument("--cpu-only", action="store_true", help="Only stress CPUs")
    parser.add_argument("--cpu-workers", type=int, default=0,
                        help="Number of CPU workers (default: all logical cores)")
    parser.add_argument("--log", type=str, default="benchmark_results.log",
                        help="Output log file path")
    args = parser.parse_args()

    # Detect GPUs
    num_gpus = 0
    if TORCH_AVAILABLE and not args.cpu_only:
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            print("[WARNING] No CUDA GPUs detected. Running CPU-only mode.")
        else:
            print(f"[✓] Found {num_gpus} GPU(s):")
            for i in range(num_gpus):
                props = torch.cuda.get_device_properties(i)
                print(f"    GPU {i}: {props.name} | {props.total_memory / 1024**3:.1f} GB VRAM")

    # Detect CPUs
    logical_cores = psutil.cpu_count(logical=True)
    num_cpu_workers = args.cpu_workers if args.cpu_workers > 0 else logical_cores
    if args.gpu_only:
        num_cpu_workers = 0
    print(f"[✓] CPU: {psutil.cpu_count(logical=False)} physical / {logical_cores} logical cores | Using {num_cpu_workers} workers")
    print(f"[✓] RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB total\n")

    if num_gpus == 0 and num_cpu_workers == 0:
        print("[ERROR] Nothing to benchmark. Exiting.")
        sys.exit(1)

    stop_event = threading.Event()

    # Shared stats dicts
    manager = multiprocessing.Manager()
    gpu_stats = manager.dict()
    cpu_stats = manager.dict()

    # Launch GPU threads
    gpu_threads = []
    for gid in range(num_gpus):
        t = threading.Thread(
            target=gpu_stress_worker,
            args=(gid, gpu_stats, stop_event, args.batch_size, args.input_size),
            daemon=True
        )
        t.start()
        gpu_threads.append(t)
        print(f"[→] GPU {gid} stress worker started")

    # Launch CPU processes
    cpu_processes = []
    cpu_stop = multiprocessing.Event()
    for i in range(num_cpu_workers):
        p = multiprocessing.Process(
            target=cpu_stress_worker,
            args=(i, cpu_stats, cpu_stop),
            daemon=True
        )
        p.start()
        cpu_processes.append(p)
    if num_cpu_workers > 0:
        print(f"[→] {num_cpu_workers} CPU worker processes started")

    print(f"\n[✓] Benchmark running. Batch size: {args.batch_size} | Input: {args.input_size}x{args.input_size}")
    print("    Press Ctrl+C to stop.\n")
    time.sleep(2)  # Let workers warm up

    try:
        monitor_and_report(
            gpu_stats=gpu_stats,
            cpu_stats=cpu_stats,
            stop_event=stop_event,
            num_gpus=num_gpus,
            num_cpu_workers=num_cpu_workers,
            duration=args.duration,
            log_file=args.log
        )
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user. Shutting down...")
        stop_event.set()

    # Cleanup
    cpu_stop.set()
    for p in cpu_processes:
        p.terminate()
        p.join(timeout=3)
    for t in gpu_threads:
        t.join(timeout=5)

    print("[✓] All workers stopped. Benchmark complete.")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
