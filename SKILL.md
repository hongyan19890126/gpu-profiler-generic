---
name: gpu-profiler-generic
description: >-
  Generic GPU profiling skill for analyzing any GPU-accelerated application
  (CUDA/OpenCL/graphics/compute/inference/training). Analyzes GPU kernel
  execution, memory transfers, API calls, and CPU/GPU timelines. Generates
  execution paths, timing breakdowns, bottleneck analysis, and optimization
  recommendations. Works with NVIDIA Nsight Systems (.nsys-rep), Nsight
  Compute (.ncu-rep), or raw nvidia-smi/nvprof data.
license: Apache-2.0
metadata:
  author: hongyan19890126
---

# Generic GPU Profiling

Profile any GPU-accelerated application regardless of domain -- training,
inference, graphics, scientific computing, or real-time rendering.

## When to Use

**Any scenario where GPU performance matters:**

- Application slower than expected on GPU
- Need to understand where time is spent (CPU vs GPU)
- Memory transfer bottlenecks (H2D/D2H/D2D)
- Kernel launch overhead investigation
- GPU idle time / underutilization
- Multi-GPU scaling issues
- API-level inefficiencies

**Works with:**
- CUDA applications (C/C++/Python)
- Deep learning frameworks (PyTorch/TensorFlow/JAX)
- Graphics engines (OpenGL/DirectX/Vulkan)
- Scientific computing (cuBLAS/cuFFT/cuDNN)
- Any program using GPU acceleration

## Requirements

| Tool | Purpose | Install |
|------|---------|---------|
| `nsys` (Nsight Systems) | Timeline + API tracing | CUDA Toolkit |
| `ncu` (Nsight Compute) | Kernel-level metrics | CUDA Toolkit |
| `nvidia-smi` | Runtime monitoring | Driver |
| `nvprof` | Legacy profiling | CUDA Toolkit |

## Analysis Framework

### 1. Execution Path Extraction

Extract the complete execution flow:

```bash
# Export execution timeline
nsys stats -r cuda_kern_exec_trace,cuda_api_trace \
    --format csv,json \
    profile.nsys-rep > timeline.csv
```

**Structure:**
```
[Host Setup] -> [H2D Transfer] -> [Kernel Launch] -> [GPU Execution] -> [D2H Transfer] -> [Host Post-Process]
     |              |                  |                 |                |              |
   CPU Time    PCIe Bandwidth    Launch Overhead   Compute Time   PCIe Bandwidth   CPU Time
```

### 2. Time Distribution Analysis

Categorize all GPU time:

```bash
nsys stats -r cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum \
    profile.nsys-rep
```

**Categories:**
- **Compute**: Kernel execution time
- **Memory**: H2D/D2H/D2D transfers
- **Launch**: API call overhead
- **Idle**: GPU gaps between operations
- **Synchronization**: cudaDeviceSynchronize, cudaStreamSynchronize

### 3. Bottleneck Detection

```bash
nsys analyze -r all profile.nsys-rep
```

**Common Bottlenecks:**

| Pattern | Signature | Impact |
|---------|-----------|--------|
| Memory-bound | High memcpy time, low compute | PCIe bandwidth limit |
| Launch-bound | High cudaLaunchKernel time | Too many small kernels |
| Compute-bound | High kernel time, low memcpy | Saturated SM utilization |
| Idle GPU | Long gaps with no kernels | CPU bottleneck |
| Sync-bound | Frequent synchronize calls | Serialization |

### 4. Kernel Analysis

```bash
# Get all kernels with detailed metrics
nsys stats -r cuda_kern_exec_sum profile.nsys-rep

# Export to CSV for complete analysis
nsys stats -r cuda_gpu_kern_sum --format csv profile.nsys-rep > all_kernels.csv
```

**Metrics per kernel:**
- Total time, avg time, min/max time
- Call count
- Queue time (launch latency)
- Grid/block configuration

**Complete Kernel Table:**
Always list ALL kernels, not just top 3-5. Many small kernels can indicate:
- Launch overhead (1000s of tiny kernels)
- Memory transfer kernels (H2D/D2H)
- Synchronization kernels (cudaDeviceSynchronize)

**Categorization:**
Group kernels by function:
- **Communication**: AllReduce, AllGather, Broadcast
- **Compute**: GEMM, Conv, Attention
- **Memory**: memcpy, memset, prefetch
- **Elementwise**: Activation, Norm, Scale
- **Other**: Indexing, Sorting, Scan

### 5. Memory Analysis

```bash
nsys stats -r cuda_gpu_mem_time_sum,cuda_api_mem_gpu_sum \
    profile.nsys-rep
```

**Track:**
- H2D (Host to Device) transfers
- D2H (Device to Host) transfers
- D2D (Device to Device) transfers
- Allocation/deallocation patterns
- Pageable vs pinned memory usage

## Profiling Workflows

### Workflow A: Quick Overview (5 minutes)

Goal: Understand high-level GPU utilization.

```bash
# 1. Profile for 10 seconds
nsys profile -t cuda,nvtx -o quick -- ./your_app

# 2. Get summary
nsys stats -r cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
    quick.nsys-rep
```

**Output:**
```
GPU Time Distribution:
- Compute: 65% [####################]
- Memory:  25% [########]
- Idle:    10% [###]

Top 30 Kernels:
1. kernel_gemm  : 45%  (120ms avg, 100 calls)
2. kernel_conv  : 15%  (30ms avg, 200 calls)
3. kernel_copy  : 10%  (5ms avg, 500 calls)
```

**Complete Kernel List:**
```bash
# Export all kernels to CSV for custom analysis
nsys stats -r cuda_gpu_kern_sum --format csv,json quick.nsys-rep > kernels.csv
```

Analyze all kernels, not just top 3. Look for:
- **High frequency, low time**: Launch overhead candidates
- **Low frequency, high time**: Optimization targets
- **Memory kernels**: H2D/D2H transfer patterns

### Workflow B: Deep Dive (30 minutes)

Goal: Detailed bottleneck analysis with optimization recommendations.

```bash
# 1. Full profile with all traces
nsys profile \
    -t cuda,nvtx,osrt,cudnn,cublas \
    --python-backtrace=cuda \
    -o deep -- ./your_app

# 2. Comprehensive analysis
nsys stats -r \
    cuda_gpu_kern_sum,\
    cuda_kern_exec_sum,\
    cuda_gpu_mem_time_sum,\
    cuda_api_sum,\
    osrt_sum \
    deep.nsys-rep

# 3. Anti-pattern detection
nsys analyze -r all deep.nsys-rep

# 4. Export for custom analysis
nsys export -t sqlite deep.nsys-rep -o deep.sqlite
```

### Workflow C: Memory-Bound Applications

Goal: Optimize data movement.

```bash
nsys profile -t cuda,nvtx --cudabacktrace=true -o mem -- ./your_app

nsys stats -r \
    cuda_gpu_mem_time_sum,\
    cuda_api_mem_gpu_sum,\
    cuda_api_mem_op_sum \
    mem.nsys-rep
```

**Focus:**
- Pageable vs pinned transfer ratio
- Asynchronous transfer overlap
- Unnecessary D2H/H2D copies
- Memory allocation patterns

### Workflow D: Kernel Optimization

Goal: Optimize specific kernels.

```bash
# Profile with Nsight Compute for kernel metrics
ncu --kernel-name kernel_name \
    --metrics sm__throughput,\
               sm__warps_active.avg.pct_of_peak_sustained_elapsed,\
               dram__throughput \
    ./your_app
```

**Metrics:**
- SM utilization (% of peak)
- Memory bandwidth (% of peak)
- Occupancy (active warps)
- Instruction mix (FMA, load, store)

## Report Generation

### Standard Report Structure

```markdown
# GPU Profiling Report - [Application Name]

## 1. Overview
- Duration: [X] seconds
- GPU: [Model] x [Count]
- GPU Utilization: [%]
- Memory Bandwidth: [% of peak]

## 2. Execution Timeline
```
[CPU] -> [H2D] -> [GPU Kernel 1] -> [GPU Kernel 2] -> [D2H] -> [CPU]
  5ms    10ms       50ms             30ms            8ms      5ms
```

## 3. Time Distribution
| Category | Time | Percentage |
|----------|------|------------|
| Compute  | 80ms | 53% |
| Memory   | 30ms | 20% |
| Launch   | 20ms | 13% |
| Idle     | 20ms | 13% |

## 4. Complete Kernel Breakdown

**Total Kernels**: [N] unique kernels
**Total Launches**: [X] launches
**Top 10 Account For**: [Y]%

### 4.1 Top Kernels

| Rank | Kernel | Time (s) | Percentage | Calls | Avg (ms) | Max (ms) |
|------|--------|----------|------------|-------|------------|----------|
| 1 | kernel_1 | 194,575.87 | 58.27% | 266,684 | 0.73 | 652.59 |
| 2 | kernel_2 | 119,405.19 | 35.76% | 134,444 | 0.89 | 652.59 |
| 3 | kernel_3 | 3,235.00 | 0.97% | 4,408 | 0.73 | 3.30 |
| 4 | kernel_4 | 2,568.45 | 0.77% | 268,888 | 0.01 | 0.02 |
| 5 | kernel_5 | 2,087.93 | 0.62% | 269,864 | 0.01 | 0.03 |
| ... | ... | ... | ... | ... | ... | ... |
| N | kernel_N | 0.01 | 0.00% | 4 | 0.00 | 0.00 |

### 4.2 Category Summary

| Category | Kernels | Total Time (s) | Percentage |
|----------|---------|----------------|------------|
| **Communication** | 4 | 198,622.45 | 59.48% |
| **GEMM/Compute** | 10 | 125,628.91 | 37.62% |
| **Attention** | 4 | 3,553.48 | 1.06% |
| **MOE/Routing** | 4 | 2,422.44 | 0.73% |
| **Normalization** | 2 | 864.65 | 0.26% |
| **Memory/Copy** | 6 | 565.09 | 0.17% |
| **Quantization** | 3 | 953.27 | 0.29% |
| **Elementwise** | 15 | 567.54 | 0.17% |
| **Other** | 26 | 734.00 | 0.22% |

### 4.3 Key Observations

1. **Concentration**: Top 2 kernels account for [X]% of total time
2. **Communication**: AllReduce operations dominate if present
3. **Launch Frequency**: High instance counts indicate small kernels
4. **Memory**: D2H transfers indicate synchronization points
5. **Multi-GPU**: NCCL kernels indicate distributed execution

## 5. Bottlenecks
1. **Memory Transfer**: 20% of time in H2D copies
   - Evidence: cudaMemcpyAsync taking 10ms per call
   - Recommendation: Use pinned memory, batch transfers

2. **Kernel Launch Overhead**: 500 small kernels
   - Evidence: 1000 launches at 0.01ms each
   - Recommendation: Fuse kernels or use CUDA Graphs

## 6. Optimization Recommendations
- [ ] P0: Use pinned memory for transfers (2x speedup)
- [ ] P1: Batch small kernels into larger ones (1.5x speedup)
- [ ] P2: Overlap compute and memory with streams (1.3x speedup)

## 7. Expected Gains
Current: 100 items/sec
Optimized: 300 items/sec (3x improvement)
```

## Optimization Patterns

### Pattern 1: Reduce Memory Transfers

**Problem**: Frequent small transfers
```
[CPU] -> [H2D: 1MB] -> [GPU: 10ms] -> [D2H: 1MB] -> [CPU]
```

**Solution**: Batch + overlap
```python
# Before: 100 x 1MB transfers
for i in range(100):
    cudaMemcpyAsync(d_buf[i], h_buf[i], 1MB)
    kernel<<<...>>>(d_buf[i])

# After: 1 x 100MB transfer + compute overlap
stream1, stream2 = cudaStreamCreate(), cudaStreamCreate()
for i in range(0, 100, 2):
    cudaMemcpyAsync(d_buf[i], h_buf[i], 1MB, stream1)
    kernel<<<..., stream1>>>(d_buf[i])
    cudaMemcpyAsync(d_buf[i+1], h_buf[i+1], 1MB, stream2)
    kernel<<<..., stream2>>>(d_buf[i+1])
```

### Pattern 2: Reduce Launch Overhead

**Problem**: 1000s of small kernel launches

**Solutions:**
1. **CUDA Graphs** (static workflows)
2. **Kernel Fusion** (combine element-wise ops)
3. **Larger grids** (fewer launches, more work per launch)

### Pattern 3: Hide Latency with Streams

```python
# Create 3 streams for overlap
streams = [torch.cuda.Stream() for _ in range(3)]

for i, batch in enumerate(dataloader):
    stream = streams[i % 3]
    with torch.cuda.stream(stream):
        # Copy + compute overlap
        batch_gpu = batch.cuda(non_blocking=True)
        output = model(batch_gpu)
```

### Pattern 4: Optimize Memory Access

**Checklist:**
- [ ] Use pinned (page-locked) host memory
- [ ] Use unified memory for simple cases
- [ ] Minimize D2H transfers (keep data on GPU)
- [ ] Use texture memory for read-only 2D access
- [ ] Coalesce global memory accesses

## Multi-GPU Analysis

### Profile Collection
```bash
nsys profile -t cuda,nvtx,mpi \
    -o profile_%q{OMPI_COMM_WORLD_RANK} \
    -- mpirun -np 4 ./app
```

### Scaling Analysis
```bash
nsys recipe cuda_gpu_time_util_map -- profile_*.nsys-rep
nsys recipe nccl_gpu_overlap_trace -- profile_*.nsys-rep
```

**Check:**
- Load balance across GPUs
- Communication/compute overlap
- Straggler detection

## Troubleshooting

| Symptom | Check | Command |
|---------|-------|---------|
| Low GPU util | CPU bottleneck | `nsys stats -r osrt_sum` |
| High latency | Memory transfers | `nsys stats -r cuda_gpu_mem_time_sum` |
| Slow kernels | Kernel efficiency | `ncu --metrics sm__throughput` |
| OOM errors | Memory allocation | `nsys stats -r cuda_api_mem_gpu_sum` |
| Hang/crash | Sync issues | `nsys stats -r cuda_api_sync_sum` |

## Safety Guidelines

1. **Never modify running production systems** -- profile on test instances
2. **Profile short durations** -- 10-30 seconds is usually sufficient
3. **Use `--capture-range=cudaProfilerApi`** to skip initialization
4. **Respect privacy** -- profiles may contain sensitive data (model weights, inputs)
5. **Disk space** -- `.nsys-rep` files can be large (GBs for long profiles)

## Integration

### CI/CD Integration
```yaml
# GitHub Actions example
- name: Profile GPU Performance
  run: |
    nsys profile -t cuda -o ci_profile -- ./test_suite
    nsys stats -r cuda_gpu_kern_sum ci_profile.nsys-rep > profile.txt
    # Fail if GPU utilization < 80%
    python check_utilization.py profile.txt --min-util=80
```

### Automated Regression Detection
```python
# Compare current vs baseline profile
import json

def compare_profiles(baseline, current):
    """Return True if current is within 10% of baseline"""
    with open(baseline) as f: base = json.load(f)
    with open(current) as f: curr = json.load(f)
    
    for kernel in base['kernels']:
        name = kernel['name']
        base_time = kernel['total_time']
        curr_time = next(k['total_time'] for k in curr['kernels'] if k['name'] == name)
        
        diff = abs(curr_time - base_time) / base_time
        if diff > 0.10:
            print(f"REGRESSION: {name} changed by {diff*100:.1f}%")
            return False
    
    return True
```

## References

- Nsight Systems Documentation: https://docs.nvidia.com/nsight-systems/
- Nsight Compute Documentation: https://docs.nvidia.com/nsight-compute/
- CUDA Best Practices Guide: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- Profiling Tools Overview: https://developer.nvidia.com/tools-overview
