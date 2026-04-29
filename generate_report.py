#!/usr/bin/env python3
"""
GPU Profiling Report Generator - Dynamic Version
Extracts REAL data from nsys-rep, no hardcoded assumptions

Usage:
    python generate_report.py <profile.nsys-rep> [output.md]
"""

import subprocess
import sys
import re
from datetime import datetime
from pathlib import Path


def run_nsys_command(cmd):
    """Run nsys command and return output"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            return None
        return result.stdout
    except Exception:
        return None


def parse_kernels(output):
    """Parse kernel statistics from nsys output"""
    if not output:
        return []
    
    kernels = []
    lines = output.strip().split('\n')
    in_table = False
    
    for line in lines:
        if 'Time (%)' in line:
            in_table = True
            continue
        if in_table and line.strip() and not line.startswith('Processing'):
            # Match lines like: " 58.3  194,575,865,215    266,684  729,612.1  ..."
            match = re.match(r'\s*(\d+\.?\d*)\s+([\d,]+)\s+([\d,]+)\s+([\d,.]+)\s+([\d,.]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,.]+)\s+(.+)', line)
            if match:
                try:
                    kernels.append({
                        'percentage': float(match.group(1)),
                        'total_time_ns': float(match.group(2).replace(',', '')),
                        'calls': int(match.group(3).replace(',', '')),
                        'avg_time_ns': float(match.group(4).replace(',', '')),
                        'name': match.group(9).strip()
                    })
                except:
                    continue
    
    return sorted(kernels, key=lambda x: x['total_time_ns'], reverse=True)


def parse_memory(output):
    """Parse memory statistics"""
    if not output:
        return []
    
    memory = []
    lines = output.strip().split('\n')
    in_table = False
    
    for line in lines:
        if 'Time (%)' in line:
            in_table = True
            continue
        if in_table and line.strip() and not line.startswith('Processing'):
            match = re.match(r'\s*(\d+\.?\d*)\s+([\d,]+)\s+([\d,]+)\s+([\d,.]+)\s+(.+)', line)
            if match:
                try:
                    memory.append({
                        'percentage': float(match.group(1)),
                        'total_time_ns': float(match.group(2).replace(',', '')),
                        'calls': int(match.group(3).replace(',', '')),
                        'name': match.group(5).strip()
                    })
                except:
                    continue
    
    return memory


def parse_api(output):
    """Parse API statistics"""
    if not output:
        return []
    
    api = []
    lines = output.strip().split('\n')
    in_table = False
    
    for line in lines:
        if 'Time (%)' in line:
            in_table = True
            continue
        if in_table and line.strip() and not line.startswith('Processing'):
            match = re.match(r'\s*(\d+\.?\d*)\s+([\d,]+)\s+([\d,]+)\s+([\d,.]+)\s+(.+)', line)
            if match:
                try:
                    api.append({
                        'percentage': float(match.group(1)),
                        'total_time_ns': float(match.group(2).replace(',', '')),
                        'calls': int(match.group(3).replace(',', '')),
                        'name': match.group(5).strip()
                    })
                except:
                    continue
    
    return api


def parse_gpu_info(output):
    """Extract GPU count and device info from nsys output"""
    if not output:
        return {'count': 1, 'devices': []}
    
    devices = set()
    for line in output.split('\n'):
        if 'Device' in line and any(x in line for x in ['0', '1', '2', '3', '4', '5', '6', '7']):
            match = re.search(r'Device\s+(\d+)', line)
            if match:
                devices.add(int(match.group(1)))
    
    return {
        'count': len(devices) if devices else 1,
        'devices': sorted(list(devices))
    }


def get_profile_duration(report_file):
    """Get actual profile duration from nsys"""
    output = run_nsys_command(f"nsys stats -r osrt_sum '{report_file}'")
    if output:
        # Try to extract duration from output
        for line in output.split('\n'):
            if 'Duration' in line or 'Total Time' in line:
                match = re.search(r'(\d+)\s*s', line)
                if match:
                    return int(match.group(1))
    return 0


def categorize_kernel(name):
    """Categorize kernel by function"""
    name_lower = name.lower()
    
    if any(x in name_lower for x in ['allreduce', 'allgather', 'nccl']):
        return 'Communication'
    elif any(x in name_lower for x in ['gemm', 'bmm', 'matmul']):
        return 'GEMM/Compute'
    elif any(x in name_lower for x in ['attention', 'fmha', 'flash']):
        return 'Attention'
    elif any(x in name_lower for x in ['moe', 'routing']):
        return 'MOE/Routing'
    elif any(x in name_lower for x in ['norm', 'rmsnorm']):
        return 'Normalization'
    elif any(x in name_lower for x in ['quantize', 'cvt_']):
        return 'Quantization'
    elif any(x in name_lower for x in ['memcpy', 'memset']):
        return 'Memory/Copy'
    elif any(x in name_lower for x in ['elementwise', 'activation']):
        return 'Elementwise'
    elif 'triton' in name_lower:
        return 'Triton'
    elif any(x in name_lower for x in ['cub', 'scan', 'sort']):
        return 'CUDA Primitives'
    else:
        return 'Other'


def generate_ascii_bar(percentage, width=40):
    """Generate ASCII progress bar"""
    filled = int(width * min(percentage, 100) / 100)
    return '[' + '#' * filled + ' ' * (width - filled) + ']'


def analyze_bottlenecks(kernels, api, memory, categories):
    """Dynamically analyze bottlenecks based on actual data"""
    bottlenecks = []
    total_time = sum(k['total_time_ns'] for k in kernels)
    
    # 1. Check for communication bottleneck
    if 'Communication' in categories:
        comm_time = categories['Communication']['time']
        comm_pct = comm_time / total_time * 100
        if comm_pct > 20:
            bottlenecks.append({
                'name': 'Communication Dominance',
                'percentage': comm_pct,
                'time': comm_time,
                'severity': '🔴 CRITICAL' if comm_pct > 50 else '🟡 MEDIUM',
                'evidence': f"Communication kernels take {comm_pct:.1f}% of GPU time",
                'impact': 'GPUs idle waiting for synchronization',
                'recommendations': [
                    'Reduce sync frequency (gradient accumulation)',
                    'Overlap communication with compute (CUDA streams)',
                    'Use faster interconnect (NVLink, InfiniBand)'
                ]
            })
    
    # 2. Check for launch overhead
    launch_apis = [a for a in api if 'launch' in a['name'].lower()]
    if launch_apis:
        total_api = sum(a['total_time_ns'] for a in api)
        launch_time = sum(a['total_time_ns'] for a in launch_apis)
        launch_pct = launch_time / total_api * 100 if total_api > 0 else 0
        if launch_pct > 30:
            bottlenecks.append({
                'name': 'Kernel Launch Overhead',
                'percentage': launch_pct,
                'time': launch_time,
                'severity': '🔴 CRITICAL' if launch_pct > 60 else '🟡 MEDIUM',
                'evidence': f"Launch APIs take {launch_pct:.1f}% of API time ({sum(a['calls'] for a in launch_apis):,} launches)",
                'impact': 'CPU bottleneck in submitting work to GPU',
                'recommendations': [
                    'Enable CUDA Graphs',
                    'Batch small kernels',
                    'Use persistent kernels'
                ]
            })
    
    # 3. Check for memory bottleneck
    if memory:
        d2h = [m for m in memory if 'Device-to-Host' in m['name']]
        if d2h:
            total_mem = sum(m['total_time_ns'] for m in memory)
            d2h_pct = d2h[0]['total_time_ns'] / total_mem * 100 if total_mem > 0 else 0
            if d2h_pct > 30:
                bottlenecks.append({
                    'name': 'Memory Transfer Overhead',
                    'percentage': d2h_pct,
                    'time': d2h[0]['total_time_ns'],
                    'severity': '🟡 MEDIUM',
                    'evidence': f"D2H transfers take {d2h_pct:.1f}% of memory time",
                    'impact': 'Pipeline stalls due to synchronization',
                    'recommendations': [
                        'Use pinned (page-locked) memory',
                        'Minimize D2H transfers',
                        'Use non_blocking transfers'
                    ]
                })
    
    # 4. Check for compute bottleneck
    if 'GEMM/Compute' in categories:
        compute_time = categories['GEMM/Compute']['time']
        compute_pct = compute_time / total_time * 100
        if compute_pct > 60:
            bottlenecks.append({
                'name': 'Compute Saturation',
                'percentage': compute_pct,
                'time': compute_time,
                'severity': '🟡 MEDIUM',
                'evidence': f"Compute kernels take {compute_pct:.1f}% of GPU time",
                'impact': 'SMs are fully utilized',
                'recommendations': [
                    'Optimize kernel efficiency with Nsight Compute',
                    'Check memory coalescing',
                    'Consider mixed precision (FP16/BF16)'
                ]
            })
    
    # 5. Check for idle GPU
    if total_time > 0:
        # Estimate idle time from gaps between kernels
        # This is a rough estimate - actual idle time needs timeline analysis
        pass
    
    return bottlenecks


def generate_report(report_file, output_file=None):
    """Generate comprehensive profiling report with REAL data"""
    
    if output_file is None:
        output_file = report_file.replace('.nsys-rep', '_report.md')
    
    print(f"Analyzing: {report_file}")
    print("This may take a few minutes...")
    
    # Extract data
    print("\n[1/5] Extracting kernel statistics...")
    kernel_output = run_nsys_command(f"nsys stats -r cuda_gpu_kern_sum '{report_file}'")
    kernels = parse_kernels(kernel_output)
    
    print("[2/5] Extracting memory statistics...")
    memory_output = run_nsys_command(f"nsys stats -r cuda_gpu_mem_time_sum '{report_file}'")
    memory = parse_memory(memory_output)
    
    print("[3/5] Extracting API statistics...")
    api_output = run_nsys_command(f"nsys stats -r cuda_api_sum '{report_file}'")
    api = parse_api(api_output)
    
    print("[4/5] Detecting GPU configuration...")
    gpu_info = parse_gpu_info(kernel_output)
    duration = get_profile_duration(report_file)
    
    print("[5/5] Generating report...")
    
    # Calculate totals
    total_kernel_time = sum(k['total_time_ns'] for k in kernels)
    total_memory_time = sum(m['total_time_ns'] for m in memory)
    total_api_time = sum(a['total_time_ns'] for a in api)
    total_calls = sum(k['calls'] for k in kernels)
    
    # Categorize kernels
    categories = {}
    for k in kernels:
        cat = categorize_kernel(k['name'])
        if cat not in categories:
            categories[cat] = {'count': 0, 'time': 0}
        categories[cat]['count'] += 1
        categories[cat]['time'] += k['total_time_ns']
    
    sorted_categories = sorted(
        categories.items(), key=lambda x: x[1]['time'], reverse=True
    )
    
    # Analyze bottlenecks
    bottlenecks = analyze_bottlenecks(kernels, api, memory, categories)
    
    # Generate report
    report = []
    
    # Header
    report.append("# GPU Profiling Report")
    report.append("")
    report.append(f"**File**: `{Path(report_file).name}`")
    report.append(f"**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Tool**: NVIDIA Nsight Systems + Generic GPU Profiler")
    report.append("")
    report.append("---")
    report.append("")
    
    # 1. Overview - with REAL data
    report.append("## 1. Overview")
    report.append("")
    report.append("| Metric | Value |")
    report.append("|--------|-------|")
    
    if duration > 0:
        report.append(f"| Total Duration | {duration}s |")
    else:
        report.append(f"| Total Kernel Time | {total_kernel_time/1e9:.2f}s |")
    
    report.append(f"| GPUs Detected | {gpu_info['count']} |")
    report.append(f"| Unique Kernels | {len(kernels)} |")
    report.append(f"| Total Kernel Launches | {total_calls:,} |")
    report.append(f"| Total GPU Time | {total_kernel_time/1e9:.2f}s |")
    report.append("")
    
    # 2. Execution Timeline - with REAL data
    report.append("## 2. Execution Timeline")
    report.append("")
    report.append("### 2.1 High-Level Flow")
    report.append("")
    report.append("```")
    
    if memory:
        h2d_time = sum(m['total_time_ns'] for m in memory if 'Host-to-Device' in m['name']) / 1e6
        d2h_time = sum(m['total_time_ns'] for m in memory if 'Device-to-Host' in m['name']) / 1e6
        report.append(f"[Host] -> [H2D: {h2d_time:.2f}ms] -> [GPU: {total_kernel_time/1e9:.0f}s] -> [D2H: {d2h_time:.2f}ms] -> [Host]")
    else:
        report.append(f"[Host] -> [H2D] -> [GPU: {total_kernel_time/1e9:.0f}s] -> [D2H] -> [Host]")
    
    report.append("```")
    report.append("")
    
    # GPU utilization - dynamic based on actual GPU count
    if gpu_info['count'] > 1:
        report.append("### 2.2 Per-GPU Execution")
        report.append("")
        report.append("| GPU | Status |")
        report.append("|-----|--------|")
        for dev in gpu_info['devices']:
            report.append(f"| Device {dev} | Active |")
        report.append("")
    
    report.append("### 2.3 GPU Execution Breakdown")
    report.append("")
    report.append(f"```")
    report.append(f"Total GPU Time: {total_kernel_time/1e9:.0f} seconds")
    report.append("")
    
    for cat_name, cat_data in sorted_categories[:6]:
        pct = cat_data['time'] / total_kernel_time * 100
        time_sec = cat_data['time'] / 1e9
        bar = generate_ascii_bar(pct, 40)
        report.append(f"{cat_name:<20} {bar}  {pct:.1f}%  {time_sec:.0f}s")
    
    report.append("```")
    report.append("")
    
    # 3. Time Distribution
    report.append("## 3. Time Distribution")
    report.append("")
    report.append("| Category | Time (s) | Percentage |")
    report.append("|----------|----------|------------|")
    
    for cat_name, cat_data in sorted_categories[:10]:
        time_sec = cat_data['time'] / 1e9
        pct = (cat_data['time'] / total_kernel_time * 100) if total_kernel_time > 0 else 0
        report.append(f"| {cat_name} | {time_sec:.2f} | {pct:.1f}% |")
    
    report.append("")
    
    # 4. Complete Kernel Breakdown
    report.append("## 4. Complete Kernel Breakdown")
    report.append("")
    report.append(f"**Total Kernels**: {len(kernels)}")
    report.append(f"**Total Launches**: {total_calls:,}")
    if len(kernels) > 0:
        top10_time = sum(k['total_time_ns'] for k in kernels[:10])
        report.append(f"**Top 10 Account For**: {top10_time/total_kernel_time*100:.1f}%")
    report.append("")
    
    report.append("### 4.1 Top Kernels")
    report.append("")
    report.append("| Rank | Kernel | Time (s) | Percentage | Calls | Avg (ms) | Category |")
    report.append("|------|--------|----------|------------|-------|------------|----------|")
    
    for i, k in enumerate(kernels[:20], 1):
        time_sec = k['total_time_ns'] / 1e9
        pct = k['percentage']
        avg_ms = k['avg_time_ns'] / 1e6
        cat = categorize_kernel(k['name'])
        name = k['name'][:50] + '...' if len(k['name']) > 50 else k['name']
        report.append(f"| {i} | {name} | {time_sec:.2f} | {pct:.2f}% | {k['calls']:,} | {avg_ms:.2f} | {cat} |")
    
    report.append("")
    
    if len(kernels) > 20:
        report.append(f"... and {len(kernels) - 20} more kernels")
        report.append("")
    
    report.append("### 4.2 Category Summary")
    report.append("")
    report.append("| Category | Kernels | Total Time (s) | Percentage |")
    report.append("|----------|---------|----------------|------------|")
    
    for cat_name, cat_data in sorted_categories:
        time_sec = cat_data['time'] / 1e9
        pct = (cat_data['time'] / total_kernel_time * 100) if total_kernel_time > 0 else 0
        report.append(f"| {cat_name} | {cat_data['count']} | {time_sec:.2f} | {pct:.2f}% |")
    
    report.append("")
    
    # 5. Memory Analysis
    if memory:
        report.append("## 5. Memory Operations")
        report.append("")
        report.append("| Direction | Time (s) | Percentage | Count |")
        report.append("|-----------|----------|------------|-------|")
        
        for m in memory[:10]:
            time_sec = m['total_time_ns'] / 1e9
            pct = m['percentage']
            report.append(f"| {m['name']} | {time_sec:.4f} | {pct:.1f}% | {m['calls']:,} |")
        
        report.append("")
    
    # 6. API Analysis
    if api:
        report.append("## 6. API Overhead")
        report.append("")
        report.append("| API | Time (s) | Percentage | Calls |")
        report.append("|-----|----------|------------|-------|")
        
        for a in api[:10]:
            time_sec = a['total_time_ns'] / 1e9
            pct = a['percentage']
            report.append(f"| {a['name']} | {time_sec:.2f} | {pct:.1f}% | {a['calls']:,} |")
        
        report.append("")
    
    # 7. Bottleneck Analysis - DYNAMIC based on actual data
    if bottlenecks:
        report.append("## 7. Bottleneck Analysis")
        report.append("")
        
        for i, b in enumerate(bottlenecks, 1):
            report.append(f"### {i}. {b['name']} ({b['percentage']:.1f}%)")
            report.append("")
            report.append(f"**Time**: {b['time']/1e9:.2f}s")
            report.append("")
            report.append("**Evidence**:")
            report.append(f"- {b['evidence']}")
            report.append("")
            report.append(f"**Impact**: {b['impact']}")
            report.append(f"**Severity**: {b['severity']}")
            report.append("")
    
    # 8. Optimization Recommendations - DYNAMIC based on bottlenecks
    report.append("## 8. Optimization Recommendations")
    report.append("")
    
    if bottlenecks:
        # P0 recommendations
        critical_bottlenecks = [b for b in bottlenecks if 'CRITICAL' in b['severity']]
        if critical_bottlenecks:
            report.append("### 🔴 P0: Critical Issues")
            report.append("")
            
            for b in critical_bottlenecks:
                report.append(f"**{b['name']}**")
                report.append("")
                for rec in b['recommendations'][:2]:
                    report.append(f"- {rec}")
                report.append("")
        
        # P1 recommendations
        medium_bottlenecks = [b for b in bottlenecks if 'MEDIUM' in b['severity']]
        if medium_bottlenecks:
            report.append("### 🟡 P1: Medium Priority")
            report.append("")
            
            for b in medium_bottlenecks:
                report.append(f"**{b['name']}**")
                report.append("")
                for rec in b['recommendations'][:2]:
                    report.append(f"- {rec}")
                report.append("")
    
    # Generic recommendations if no specific bottlenecks
    if not bottlenecks:
        report.append("No major bottlenecks detected. General optimizations:")
        report.append("")
        report.append("- Profile with Nsight Compute for kernel-level optimization")
        report.append("- Check memory access patterns")
        report.append("- Consider mixed precision (FP16/BF16)")
        report.append("")
    
    # 9. Action Items
    report.append("## 9. Action Items")
    report.append("")
    report.append("- [ ] Review bottleneck analysis above")
    report.append("- [ ] Implement P0 recommendations first")
    report.append("- [ ] Profile again after optimizations")
    report.append("- [ ] Compare before/after performance")
    report.append("")
    
    # Footer
    report.append("---")
    report.append("*Generated by GPU Profiler Generic Skill*")
    report.append("*Repository: https://github.com/hongyan19890126/gpu-profiler-generic*")
    
    # Write report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"\n✅ Report generated: {output_file}")
    print(f"   Kernels: {len(kernels)}")
    print(f"   Categories: {len(categories)}")
    print(f"   Bottlenecks: {len(bottlenecks)}")
    print(f"   Total time: {total_kernel_time/1e9:.2f}s")


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_report.py <profile.nsys-rep> [output.md]")
        print("\nExample:")
        print("  python generate_report.py layerwise_profile_v2.nsys-rep")
        sys.exit(1)
    
    report_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(report_file).exists():
        print(f"Error: File not found: {report_file}")
        sys.exit(1)
    
    generate_report(report_file, output_file)


if __name__ == "__main__":
    main()
