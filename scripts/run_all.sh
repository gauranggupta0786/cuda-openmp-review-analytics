#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_all.sh — Compile and run all programs
# Usage: cd scripts && bash run_all.sh
# Prerequisite: reviews.csv and vader_lexicon.txt must exist (see data/README.md)
# ─────────────────────────────────────────────────────────────────────────────

set -e   # exit immediately on any error
cd "$(dirname "$0")"   # always run from scripts/ regardless of where called from

SRC=../src
ROOT=..

echo "════════════════════════════════════════"
echo " Compiling all programs"
echo "════════════════════════════════════════"

echo "[1/5] CUDA baseline..."
nvcc -std=c++14 -Wno-deprecated-gpu-targets $SRC/cuda_toprated.cu -o $ROOT/baseline

echo "[2/5] CUDA optimized..."
nvcc -std=c++14 -Wno-deprecated-gpu-targets $SRC/cuda_toprated_optimized.cu -o $ROOT/opt

echo "[3/5] CUDA sentiment analysis..."
nvcc -Wno-deprecated-gpu-targets $SRC/cuda_reviewanalysis.cu -o $ROOT/sentiment

echo "[4/5] Sequential C++..."
g++ $SRC/c_elaborate.c -o $ROOT/seq

echo "[5/5] OpenMP..."
g++ -fopenmp $SRC/c_elaborate_openmp.c -o $ROOT/omp

echo ""
echo "════════════════════════════════════════"
echo " Running CUDA Programs"
echo "════════════════════════════════════════"

echo ""
echo "─── Baseline ───"
cd $ROOT && ./baseline

echo ""
echo "─── Optimized ───"
./opt

echo ""
echo "─── Sentiment Analysis ───"
./sentiment

echo ""
echo "════════════════════════════════════════"
echo " Running CPU Programs"
echo "════════════════════════════════════════"

echo ""
echo "─── Sequential ───"
./seq

echo ""
echo "─── OpenMP (8 threads) ───"
OMP_NUM_THREADS=8 ./omp

echo ""
echo "All done. See results/ for output files."
