
### The HeavyModel Class

```python
class HeavyModel(nn.Module):
    def __init__(self, input_size=224, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Linear(4096, num_classes)
        )
```
This class creates a fake neural network that mimics the shape of a real vision model like VGG or early ResNet. Instead of creating normal matrix multiplication benchmark this will test the GPU the same way a real training job would. It hits convolution layers stressing the tensor cores and memory bandwidth, batch normalization, and dense linear layers (pure matrix multiply operations).

The architecture goes 3 channels 64, 128,256,512 feature maps through convolutions, then collapses spatially with AdaptiveAvgPool down to a 7x7 grid. Then flattens and runs through three linear layers ending at 1000 output classes. This will fill enough VRAM and generate real compute load without being so lare it immediately OOMs on 2080s.

The input_size parameter (defaults 224) controls the spatial dimensions of the fake images we feed. 

---

### The GPU Stress Worker

```python
def gpu_stress_worker(gpu_id, stats, stop_event, batch_size, input_size):
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    model = HeavyModel(input_size=input_size).to(device)
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
```

Each GPU gets its own completely independent worker function. `torch.cuda.set_device(device)` is critical — it pins this thread to one specific GPU so CUDA operations don't accidentally land on the wrong device when multiple threads are running simultaneously.

`model.train()` sets the model into training mode, which matters because BatchNorm behaves differently in training vs inference mode. In training mode it computes mean and variance from the current batch, which is more expensive and more realistic.

The `GradScaler` is part of mixed precision training. When you use `torch.cuda.amp.autocast()`, PyTorch automatically runs some operations in float16 instead of float32. Float16 math is roughly 2x faster on modern NVIDIA GPUs and uses half the VRAM, but it can cause numerical instability (gradients underflowing to zero). The GradScaler solves this by scaling the loss up before backprop and scaling the gradients back down afterward, keeping them in a numerically stable range. 

Adam optimizer is used because it maintains per-parameter momentum state in VRAM, which adds meaningful memory pressure beyond just the model weights.


```python
while not stop_event.is_set():
    inputs = torch.randn(batch_size, 3, input_size, input_size, device=device)
    targets = torch.randint(0, 1000, (batch_size,), device=device)

    optimizer.zero_grad(set_to_none=True)

    with torch.cuda.amp.autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

This is the core loop — it runs indefinitely until the stop event is set. Every iteration it generates a fresh batch of random fake images directly on the GPU (not on CPU then transferred — `device=device` in the `torch.randn` call is important for speed), does a full forward pass, computes loss, runs backpropagation, and updates weights. This is a complete simulated training step.


```python
except torch.cuda.OutOfMemoryError:
    batch_size = batch_size // 2
    torch.cuda.empty_cache()
```

Rather than crashing on OOM, the worker halves the batch size and clears the cache. This lets the benchmark continue even if you were too aggressive with your initial batch size setting, and it tells you exactly where the VRAM ceiling is.

---

## The CPU Stress Worker

```python
def cpu_stress_worker(worker_id, stats, stop_event):
    size = 512
    a = [float(i % 100) for i in range(size * size)]
    b = [float((i + 1) % 100) for i in range(size * size)]

    while not stop_event.is_set():
        result = 0.0
        for i in range(0, len(a), 64):
            chunk = a[i:i+64]
            bchunk = b[i:i+64]
            result += sum(x * y + math.sqrt(abs(x - y) + 1e-9) for x, y in zip(chunk, bchunk))
```

This is intentionally written in pure Python with no numpy or external dependencies. The reason is portability — you can run this even on a machine where numpy isn't installed, and it genuinely stresses the CPU rather than offloading the work to an optimized C library.

The math per iteration combines multiplication, subtraction, absolute value, and square root. Square root is specifically included because it's a slow floating point operation that can't be easily vectorized by the compiler, which means it reliably consumes real CPU cycles rather than being optimized away.

The 64-element chunk size is chosen to fit comfortably in L1 cache while still requiring real memory access patterns. The 512×512 total array size is large enough to create genuine memory pressure but not so large it spills into main RAM and becomes a memory bandwidth test instead of a compute test.


---

## The Monitoring Function

```python
def monitor_and_report(gpu_stats, cpu_stats, stop_event, ...):
    while not stop_event.is_set():
        os.system("clear")
        nvml_data = get_gpu_stats_nvml()
        ...
        time.sleep(2)
```

The monitor runs in the main thread. Every 2 seconds it clears the terminal and redraws the dashboard — that's the "live" effect. It pulls two different data sources simultaneously: the `gpu_stats` shared dict that each GPU worker updates after every iteration (PyTorch-level data like throughput and allocated memory), and pynvml for hardware-level data (temperature, total utilization percentage, total VRAM from the driver's perspective).

---

## The Argument Parser & Main Function
```bash
# Light load — good starting point on your desktop
python benchmark_stress.py --batch-size 32 --duration 300

# Medium load
python benchmark_stress.py --batch-size 64 --duration 300

# Heavy — pushes VRAM hard, good for server testing
python benchmark_stress.py --batch-size 128 --duration 300

# Maximum pressure — combine large batch AND large input size
python benchmark_stress.py --batch-size 128 --input-size 512 --duration 300

# Only stress the GPU, leave CPU alone
python benchmark_stress.py --gpu-only --duration 300

# Only stress the CPU, ignore GPU entirely
python benchmark_stress.py --cpu-only --duration 300

```


---
