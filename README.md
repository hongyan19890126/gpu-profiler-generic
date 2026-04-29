# GPU Profiler Generic

[![GitHub stars](https://img.shields.io/github/stars/hongyan19890126/gpu-profiler-generic?style=social)](https://github.com/hongyan19890126/gpu-profiler-generic/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/hongyan19890126/gpu-profiler-generic?style=social)](https://github.com/hongyan19890126/gpu-profiler-generic/network/members)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Skill](https://img.shields.io/badge/Skill-OpenCode-purple.svg)](https://github.com/features/copilot)

A generic GPU profiling skill for analyzing any GPU-accelerated application (CUDA/OpenCL/graphics/compute/inference/training). Analyzes GPU kernel execution, memory transfers, API calls, and CPU/GPU timelines.

> **Quick Install**: `npx skills add hongyan19890126/gpu-profiler-generic -g -y`

## Features

- **Scene-agnostic**: Works for training, inference, graphics, scientific computing, or any GPU workload
- **Complete kernel analysis**: Lists ALL kernels (not just top 3-5) with detailed metrics
- **Execution path extraction**: Maps complete CPU -> GPU -> CPU flow
- **Bottleneck detection**: Identifies memory-bound, launch-bound, compute-bound patterns
- **Optimization recommendations**: Provides actionable optimization strategies

## Installation

```bash
npx skills add hongyan19890126/gpu-profiler-generic
```

Or clone manually:
```bash
git clone https://github.com/hongyan19890126/gpu-profiler-generic.git
```

## Requirements

- NVIDIA Nsight Systems (`nsys`)
- NVIDIA Nsight Compute (`ncu`)
- CUDA Toolkit (>= 11.0)

## Quick Start

### 1. Profile your application
```bash
nsys profile -t cuda,nvtx -o profile -- ./your_app
```

### 2. Generate report
```bash
nsys stats -r cuda_gpu_kern_sum,cuda_gpu_mem_time_sum profile.nsys-rep
```

### 3. Analyze all kernels
```bash
nsys stats -r cuda_gpu_kern_sum --format csv profile.nsys-rep > all_kernels.csv
```

## Analysis Framework

### Execution Path
```
[Host Setup] -> [H2D Transfer] -> [Kernel Launch] -> [GPU Execution] -> [D2H Transfer] -> [Host Post-Process]
```

### Time Distribution Categories
- **Compute**: Kernel execution time
- **Memory**: H2D/D2H/D2D transfers
- **Launch**: API call overhead
- **Idle**: GPU gaps between operations
- **Synchronization**: cudaDeviceSynchronize calls

### Bottleneck Patterns
| Pattern | Signature | Impact |
|---------|-----------|--------|
| Memory-bound | High memcpy time, low compute | PCIe bandwidth limit |
| Launch-bound | High cudaLaunchKernel time | Too many small kernels |
| Compute-bound | High kernel time, low memcpy | Saturated SM utilization |
| Idle GPU | Long gaps with no kernels | CPU bottleneck |
| Sync-bound | Frequent synchronize calls | Serialization |

## Report Template

The skill generates comprehensive reports including:

1. **Overview**: Duration, GPUs, utilization
2. **Execution Timeline**: Visual flow representation
3. **Complete Kernel List**: ALL kernels with time/percentage/calls
4. **Category Summary**: Grouped by function (Compute/Communication/Memory)
5. **Bottleneck Analysis**: Identified issues with evidence
6. **Optimization Recommendations**: Prioritized action items

### Sample Output
```markdown
# GPU Profiling Report

## Complete Kernel Breakdown (74 Kernels)
| Rank | Kernel | Time (s) | Percentage | Calls | Avg (ms) |
|------|--------|----------|------------|-------|----------|
| 1 | allreduce_fusion_kernel | 194,575.87 | 58.27% | 266,684 | 0.73 |
| 2 | fused_a_gemm_kernel | 119,405.19 | 35.76% | 134,444 | 0.89 |
| ... | ... | ... | ... | ... | ... |
| 74 | DeviceCompactInitKernel | 0.01 | 0.00% | 4 | 0.00 |

## Category Summary
| Category | Kernels | Total Time (s) | Percentage |
|----------|---------|----------------|------------|
| Communication | 4 | 198,622.45 | 59.48% |
| GEMM/Compute | 10 | 125,628.91 | 37.62% |
| ... | ... | ... | ... |
```

## Workflows

### Quick Overview (5 minutes)
```bash
nsys profile -t cuda,nvtx -o quick -- ./app
nsys stats -r cuda_gpu_kern_sum,cuda_gpu_mem_time_sum quick.nsys-rep
```

### Deep Dive (30 minutes)
```bash
nsys profile -t cuda,nvtx,osrt,cudnn,cublas -o deep -- ./app
nsys stats -r cuda_gpu_kern_sum,cuda_kern_exec_sum,cuda_gpu_mem_time_sum deep.nsys-rep
nsys analyze -r all deep.nsys-rep
```

### Memory-Bound Analysis
```bash
nsys profile -t cuda,nvtx --cudabacktrace=true -o mem -- ./app
nsys stats -r cuda_gpu_mem_time_sum,cuda_api_mem_gpu_sum mem.nsys-rep
```

## Optimization Patterns

### 1. Reduce Memory Transfers
```python
# Before: 100 x 1MB transfers
for i in range(100):
    cudaMemcpyAsync(d_buf[i], h_buf[i], 1MB)
    kernel<<<...>>>(d_buf[i])

# After: Batch + overlap with streams
stream1, stream2 = cudaStreamCreate(), cudaStreamCreate()
for i in range(0, 100, 2):
    cudaMemcpyAsync(d_buf[i], h_buf[i], 1MB, stream1)
    kernel<<<..., stream1>>>(d_buf[i])
    cudaMemcpyAsync(d_buf[i+1], h_buf[i+1], 1MB, stream2)
    kernel<<<..., stream2>>>(d_buf[i+1])
```

### 2. Reduce Launch Overhead
- **CUDA Graphs**: Capture and replay static execution
- **Kernel Fusion**: Combine element-wise operations
- **Larger grids**: More work per launch

### 3. Hide Latency with Streams
```python
streams = [torch.cuda.Stream() for _ in range(3)]
for i, batch in enumerate(dataloader):
    stream = streams[i % 3]
    with torch.cuda.stream(stream):
        batch_gpu = batch.cuda(non_blocking=True)
        output = model(batch_gpu)
```

### 4. Optimize Memory Access
- [ ] Use pinned (page-locked) host memory
- [ ] Minimize D2H transfers (keep data on GPU)
- [ ] Use unified memory for simple cases
- [ ] Coalesce global memory accesses

## Multi-GPU Analysis

```bash
# Profile all ranks
nsys profile -t cuda,nvtx,mpi -o profile_%q{RANK} -- mpirun -np 4 ./app

# Analyze scaling
nsys recipe cuda_gpu_time_util_map -- profile_*.nsys-rep
nsys recipe nccl_gpu_overlap_trace -- profile_*.nsys-rep
```

## Safety Guidelines

1. **Never profile production systems** -- use test instances
2. **Profile short durations** -- 10-30 seconds is usually sufficient
3. **Use `--capture-range=cudaProfilerApi`** to skip initialization
4. **Respect privacy** -- profiles may contain sensitive data
5. **Disk space** -- `.nsys-rep` files can be large (GBs)

## Troubleshooting

| Symptom | Check | Command |
|---------|-------|---------|
| Low GPU util | CPU bottleneck | `nsys stats -r osrt_sum` |
| High latency | Memory transfers | `nsys stats -r cuda_gpu_mem_time_sum` |
| Slow kernels | Kernel efficiency | `ncu --metrics sm__throughput` |
| OOM errors | Memory allocation | `nsys stats -r cuda_api_mem_gpu_sum` |
| Hang/crash | Sync issues | `nsys stats -r cuda_api_sync_sum` |

## CI/CD Integration

```yaml
# GitHub Actions example
- name: Profile GPU Performance
  run: |
    nsys profile -t cuda -o ci_profile -- ./test_suite
    nsys stats -r cuda_gpu_kern_sum ci_profile.nsys-rep > profile.txt
    # Fail if GPU utilization < 80%
    python check_utilization.py profile.txt --min-util=80
```

## References

- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

## License

Apache-2.0

## Author

hongyan19890126
