#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# benchmark.sh — Full performance benchmark
# Usage: cd scripts && bash benchmark.sh
# Note: Compile first using run_all.sh
# ─────────────────────────────────────────────────────────────────────────────

cd "$(dirname "$0")/.."   # run from repo root

echo "════════════════════════════════════════"
echo " CUDA Benchmarks (kernel time via cudaEventRecord)"
echo "════════════════════════════════════════"

echo ""
echo "─── Baseline (3 runs) ───"
for i in 1 2 3; do
    echo "Run $i:"
    ./baseline 2>&1 | grep "Baseline Time"
done

echo ""
echo "─── Optimized (3 runs) ───"
for i in 1 2 3; do
    echo "Run $i:"
    ./opt 2>&1 | grep "Optimized Time"
done

echo ""
echo "════════════════════════════════════════"
echo " CUDA Profiling (Nsight Systems)"
echo "════════════════════════════════════════"

echo ""
echo "─── Profiling baseline ───"
nsys profile --output=results/nsys_baseline ./baseline

echo ""
echo "─── Profiling optimized ───"
nsys profile --output=results/nsys_optimized ./opt

echo ""
echo "════════════════════════════════════════"
echo " Sequential Baseline"
echo "════════════════════════════════════════"

echo ""
time ./seq

echo ""
echo "════════════════════════════════════════"
echo " OpenMP Thread Scaling"
echo "════════════════════════════════════════"

for T in 2 4 8 16; do
    echo ""
    echo "─── Threads = $T ───"
    time OMP_NUM_THREADS=$T ./omp 2>&1 | grep -E "Threads|real"
done

echo ""
echo "════════════════════════════════════════"
echo " Benchmark Complete"
echo "════════════════════════════════════════"
