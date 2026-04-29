#!/usr/bin/env python3
"""
GPU Profiling Report Generator
Automatically generate comprehensive profiling report from .nsys-rep file

Usage:
    python generate_report.py <profile.nsys-rep> [output.md]

Example:
    python generate_report.py layerwise_profile_v2.nsys-rep
    python generate_report.py profile.nsys-rep my_report.md
"""

import subprocess
import json
import sys
import re
from datetime import datetime
from pathlib import Path


def run_nsys_command(cmd):
    """Run nsys command and return output"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode != 0:
            print(f"Warning: Command failed: {cmd}")
            print(f"Error: {result.stderr}")
            return None
        return result.stdout
    except subprocess.TimeoutExpired:
        print(f"Error: Command timed out: {cmd}")
        return None
    except Exception as e:
        print(f"Error running command: {e}")
        return None


def parse_nsys_stats(output, report_type):
    """Parse nsys stats output into structured data"""
    if not output:
        return []
    
    lines = output.strip().split('\n')
    data = []
    
    # Find header line
    header_idx = -1
    for i, line in enumerate(lines):
        if 'Time (%)' in line or 'Time(%)' in line:
            header_idx = i
            break
    
    if header_idx == -1:
        return []
    
    # Parse data lines (skip header and separator)
    for line in lines[header_idx + 2:]:
        if not line.strip() or line.startswith('Processing'):
            continue
        
        # Try to parse columns
        parts = line.split()
        if len(parts) < 5:
            continue
        
        try:
            # Extract percentage (first column)
            pct = float(parts[0])
            
            # Extract total time (second column, handle commas)
            time_str = parts[1].replace(',', '')
            total_time = float(time_str)
            
            # Extract instances/calls (third column)
            calls_str = parts[2].replace(',', '')
            calls = int(calls_str)
            
            # Extract average time (fourth column)
            avg_str = parts[3].replace(',', '')
            avg_time = float(avg_str)
            
            # Extract name (remaining columns)
            name = ' '.join(parts[7:]) if len(parts) > 7 else ' '.join(parts[4:])
            name = name.strip()
            
            data.append({
                'name': name,
                'percentage': pct,
                'total_time_ns': total_time,
                'calls': calls,
                'avg_time_ns': avg_time
            })
        except (ValueError, IndexError):
            continue
    
    return data


def categorize_kernel(name):
    """Categorize kernel by function"""
    name_lower = name.lower()
    
    if 'allreduce' in name_lower or 'allgather' in name_lower or 'nccl' in name_lower:
        return 'Communication'
    elif 'gemm' in name_lower or 'bmm' in name_lower or 'matmul' in name_lower:
        return 'GEMM/Compute'
    elif 'attention' in name_lower or 'fmha' in name_lower or 'flash' in name_lower:
        return 'Attention'
    elif 'moe' in name_lower or 'routing' in name_lower:
        return 'MOE/Routing'
    elif 'norm' in name_lower or 'layernorm' in name_lower or 'rmsnorm' in name_lower:
        return 'Normalization'
    elif 'quantize' in name_lower or 'cvt_' in name_lower:
        return 'Quantization'
    elif 'memcpy' in name_lower or 'memset' in name_lower:
        return 'Memory/Copy'
    elif 'elementwise' in name_lower or 'activation' in name_lower or 'act_' in name_lower:
        return 'Elementwise'
    elif 'triton' in name_lower:
        return 'Triton'
    elif 'cub' in name_lower or 'scan' in name_lower or 'sort' in name_lower:
        return 'CUDA Primitives'
    else:
        return 'Other'


def generate_report(report_file, output_file=None):
    """Generate comprehensive profiling report"""
    
    if output_file is None:
        output_file = report_file.replace('.nsys-rep', '_report.md')
    
    print(f"Analyzing: {report_file}")
    print("This may take a few minutes...")
    
    # Extract data
    print("\n[1/4] Extracting kernel statistics...")
    kernel_output = run_nsys_command(
        f"nsys stats -r cuda_gpu_kern_sum '{report_file}'"
    )
    kernels = parse_nsys_stats(kernel_output, "cuda_gpu_kern_sum")
    
    print("[2/4] Extracting memory statistics...")
    memory_output = run_nsys_command(
        f"nsys stats -r cuda_gpu_mem_time_sum '{report_file}'"
    )
    memory = parse_nsys_stats(memory_output, "cuda_gpu_mem_time_sum")
    
    print("[3/4] Extracting API statistics...")
    api_output = run_nsys_command(
        f"nsys stats -r cuda_api_sum '{report_file}'"
    )
    api = parse_nsys_stats(api_output, "cuda_api_sum")
    
    print("[4/4] Generating report...")
    
    # Calculate totals
    total_kernel_time = sum(k['total_time_ns'] for k in kernels)
    total_memory_time = sum(m['total_time_ns'] for m in memory)
    total_api_time = sum(a['total_time_ns'] for a in api)
    
    # Categorize kernels
    categories = {}
    for k in kernels:
        cat = categorize_kernel(k['name'])
        if cat not in categories:
            categories[cat] = {'count': 0, 'time': 0, 'kernels': []}
        categories[cat]['count'] += 1
        categories[cat]['time'] += k['total_time_ns']
        categories[cat]['kernels'].append(k)
    
    # Sort categories by time
    sorted_categories = sorted(
        categories.items(),
        key=lambda x: x[1]['time'],
        reverse=True
    )
    
    # Generate markdown report
    report = []
    report.append("# GPU Profiling Report")
    report.append("")
    report.append(f"**File**: `{Path(report_file).name}`")
    report.append(f"**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Tool**: NVIDIA Nsight Systems + Generic GPU Profiler")
    report.append("")
    
    # Overview
    report.append("## 1. Overview")
    report.append("")
    report.append(f"| Metric | Value |")
    report.append(f"|--------|-------|")
    report.append(f"| Total Kernels | {len(kernels)} |")
    report.append(f"| Total Kernel Time | {total_kernel_time/1e9:.2f}s |")
    report.append(f"| Total Memory Time | {total_memory_time/1e9:.2f}s |")
    report.append(f"| Total API Time | {total_api_time/1e9:.2f}s |")
    report.append("")
    
    # Execution Timeline
    report.append("## 2. Execution Timeline")
    report.append("")
    report.append("```")
    report.append("[Host] -> [H2D] -> [GPU Execution] -> [D2H] -> [Host]")
    report.append("```")
    report.append("")
    
    # Time Distribution
    report.append("## 3. Time Distribution")
    report.append("")
    report.append("| Category | Time (s) | Percentage |")
    report.append("|----------|----------|------------|")
    
    for cat_name, cat_data in sorted_categories[:10]:
        time_sec = cat_data['time'] / 1e9
        pct = (cat_data['time'] / total_kernel_time * 100) if total_kernel_time > 0 else 0
        report.append(f"| {cat_name} | {time_sec:.2f} | {pct:.1f}% |")
    
    report.append("")
    
    # Complete Kernel Breakdown
    report.append("## 4. Complete Kernel Breakdown")
    report.append("")
    report.append(f"**Total Kernels**: {len(kernels)}")
    report.append(f"**Total Launches**: {sum(k['calls'] for k in kernels):,}")
    if len(kernels) > 0:
        top10_time = sum(k['total_time_ns'] for k in kernels[:10])
        report.append(f"**Top 10 Account For**: {top10_time/total_kernel_time*100:.1f}%")
    report.append("")
    
    # Top kernels table
    report.append("### 4.1 All Kernels (Sorted by Time)")
    report.append("")
    report.append("| Rank | Kernel | Time (s) | Percentage | Calls | Avg (ms) | Category |")
    report.append("|------|--------|----------|------------|-------|------------|----------|")
    
    for i, k in enumerate(kernels, 1):
        time_sec = k['total_time_ns'] / 1e9
        pct = k['percentage']
        avg_ms = k['avg_time_ns'] / 1e6
        cat = categorize_kernel(k['name'])
        # Truncate long names
        name = k['name'][:60] + '...' if len(k['name']) > 60 else k['name']
        report.append(f"| {i} | {name} | {time_sec:.2f} | {pct:.2f}% | {k['calls']:,} | {avg_ms:.2f} | {cat} |")
    
    report.append("")
    
    # Category Summary
    report.append("### 4.2 Category Summary")
    report.append("")
    report.append("| Category | Kernels | Total Time (s) | Percentage |")
    report.append("|----------|---------|----------------|------------|")
    
    for cat_name, cat_data in sorted_categories:
        time_sec = cat_data['time'] / 1e9
        pct = (cat_data['time'] / total_kernel_time * 100) if total_kernel_time > 0 else 0
        report.append(f"| {cat_name} | {cat_data['count']} | {time_sec:.2f} | {pct:.2f}% |")
    
    report.append("")
    
    # Memory Analysis
    if memory:
        report.append("## 5. Memory Operations")
        report.append("")
        report.append("| Direction | Time (s) | Percentage | Count |")
        report.append("|-----------|----------|------------|-------|")
        
        for m in memory[:10]:
            time_sec = m['total_time_ns'] / 1e9
            pct = m['percentage']
            report.append(f"| {m['name']} | {time_sec:.2f} | {pct:.1f}% | {m['calls']:,} |")
        
        report.append("")
    
    # API Analysis
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
    
    # Bottleneck Analysis
    report.append("## 7. Bottleneck Analysis")
    report.append("")
    
    # Find top bottleneck
    if sorted_categories:
        top_cat, top_data = sorted_categories[0]
        top_pct = top_data['time'] / total_kernel_time * 100
        report.append(f"1. **{top_cat} Dominance ({top_pct:.1f}%)**")
        report.append(f"   - Time: {top_data['time']/1e9:.2f}s")
        report.append(f"   - Kernels: {top_data['count']}")
        if top_cat == 'Communication':
            report.append(f"   - Impact: GPUs idle waiting for synchronization")
            report.append(f"   - Recommendation: Reduce sync frequency, overlap communication")
        elif top_cat == 'GEMM/Compute':
            report.append(f"   - Impact: Saturated compute units")
            report.append(f"   - Recommendation: Optimize kernel efficiency")
        report.append("")
    
    # Check launch overhead
    launch_apis = [a for a in api if 'launch' in a['name'].lower()]
    if launch_apis:
        launch_time = sum(a['total_time_ns'] for a in launch_apis)
        launch_pct = launch_time / total_api_time * 100 if total_api_time > 0 else 0
        report.append(f"2. **Launch Overhead ({launch_pct:.1f}%)**")
        report.append(f"   - APIs: {', '.join(a['name'] for a in launch_apis[:3])}")
        report.append(f"   - Recommendation: Use CUDA Graphs, batch kernels")
        report.append("")
    
    # Optimization Recommendations
    report.append("## 8. Optimization Recommendations")
    report.append("")
    
    if sorted_categories and sorted_categories[0][0] == 'Communication':
        report.append("- [ ] P0: Reduce communication frequency (gradient accumulation)")
        report.append("- [ ] P0: Overlap communication with compute (CUDA streams)")
    
    if launch_apis and sum(a['total_time_ns'] for a in launch_apis) / total_api_time * 100 > 30:
        report.append("- [ ] P0: Enable CUDA Graphs to reduce launch overhead")
    
    report.append("- [ ] P1: Use pinned memory for transfers")
    report.append("- [ ] P1: Batch small kernels into larger ones")
    report.append("- [ ] P2: Profile again after optimizations")
    report.append("")
    
    # Footer
    report.append("---")
    report.append("*Generated by GPU Profiler Generic Skill*")
    report.append(f"*Repository: https://github.com/hongyan19890126/gpu-profiler-generic*")
    
    # Write report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"\n✅ Report generated: {output_file}")
    print(f"   Kernels analyzed: {len(kernels)}")
    print(f"   Categories: {len(categories)}")
    print(f"   Total time: {total_kernel_time/1e9:.2f}s")


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_report.py <profile.nsys-rep> [output.md]")
        print("\nExample:")
        print("  python generate_report.py layerwise_profile_v2.nsys-rep")
        print("  python generate_report.py profile.nsys-rep my_report.md")
        sys.exit(1)
    
    report_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(report_file).exists():
        print(f"Error: File not found: {report_file}")
        sys.exit(1)
    
    generate_report(report_file, output_file)


if __name__ == "__main__":
    main()
