#!/usr/bin/env python3
"""Compare V0 and V4.9 large eval results on the 3 demo benchmarks.
Run after both eval jobs complete:
  python3 demo/compare_demo.py
"""
import json, os, sys

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))

# Load baseline times
with open(os.path.join(DEMO_DIR, "bench_eval_base.json")) as f:
    baselines = json.load(f)

EVAL_DIR = os.path.join(DEMO_DIR, "eval")

# Load eval results
results = {}
for ckpt, label in [("checkpoint_4400_v0.json", "V0"), ("checkpoint_3200_v49.json", "V4.9 large")]:
    ckpt_path = os.path.join(EVAL_DIR, ckpt)
    if not os.path.exists(ckpt_path):
        print(f"WARNING: {ckpt_path} not found")
        results[label] = None
    else:
        with open(ckpt_path) as f:
            results[label] = json.load(f)

if all(v is None for v in results.values()):
    print("No results found. Run eval first:")
    print("  python3 demo/eval_demo.py --config demo/eval_v49.json --checkpoint 3200")
    sys.exit(1)

# Print table
print("=" * 80)
print("Demo Comparison: V0 (ckpt 4400) vs V4.9 Large (ckpt 3200)")
print("=" * 80)
print(f"{'Benchmark':<50} {'Base(ms)':<10} {'V0(ms)':<10} {'V0 sp':<8} {'V4.9(ms)':<10} {'V4.9 sp':<8} {'Ratio':<8}")
print("-" * 80)

for bench in sorted(baselines):
    base_ns = baselines[bench]
    base_ms = base_ns / 1e6

    v0_ns = results.get("V0", {}).get(bench) if results.get("V0") else None
    v49_ns = results.get("V4.9 large", {}).get(bench) if results.get("V4.9 large") else None

    v0_s = base_ns / v0_ns if v0_ns and v0_ns > 0 else None
    v49_s = base_ns / v49_ns if v49_ns and v49_ns > 0 else None

    v0_ms = v0_ns / 1e6 if v0_ns else None
    v49_ms = v49_ns / 1e6 if v49_ns else None

    ratio = v49_s / v0_s if (v0_s and v49_s and v0_s > 0) else None

    v0_str = f"{v0_ms:.2f}" if v0_ms else "FAIL"
    v0_sp = f"{v0_s:.2f}x" if v0_s else "N/A"
    v49_str = f"{v49_ms:.2f}" if v49_ms else "FAIL"
    v49_sp = f"{v49_s:.2f}x" if v49_s else "N/A"
    ratio_str = f"{ratio:.1f}x" if ratio else "N/A"

    print(f"  {bench:<48} {base_ms:<10.2f} {v0_str:<10} {v0_sp:<8} {v49_str:<10} {v49_sp:<8} {ratio_str:<8}")

print("-" * 80)

v0_speedups = []
v49_speedups = []
for bench in sorted(baselines):
    base_ns = baselines[bench]
    v0_ns = results.get("V0", {}).get(bench) if results.get("V0") else None
    v49_ns = results.get("V4.9 large", {}).get(bench) if results.get("V4.9 large") else None
    if v0_ns and v0_ns > 0:
        v0_speedups.append(base_ns / v0_ns)
    if v49_ns and v49_ns > 0:
        v49_speedups.append(base_ns / v49_ns)

if v0_speedups:
    print(f"  V0 avg speedup:          {sum(v0_speedups)/len(v0_speedups):.2f}x")
if v49_speedups:
    print(f"  V4.9 large avg speedup:  {sum(v49_speedups)/len(v49_speedups):.2f}x")
