# Project Implementation Guide

**Project:** Amazon Movies & TV Review Analysis
**Author:** Gaurang Gupta (2023A7PS0648P)
**Hardware:** RTX 3050 (4GB), CUDA 12.9, Ubuntu 24.04
**Date:** April 2026

---

## Environment

```bash
nvcc --version       # CUDA 12.9, Build cuda_12.9.r12.9/compiler.36037853_0
nvidia-smi           # RTX 3050, Driver 581.83, CUDA 13.0
```

---

## Part 1 — Setup

```bash
sudo apt update
sudo apt install build-essential python3 python3-pip -y
```

---

## Part 2 — Dataset

```bash
wget https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz
gunzip reviews_Movies_and_TV_5.json.gz
python3 src/preprocess.py        # → reviews.csv (1,697,533 rows)
```

---

## Part 3 — CUDA: Top Rated Movies

### Baseline

```bash
nvcc -std=c++14 src/cuda_toprated.cu -o baseline
./baseline
```

**Output:**
```
Total reviews: 1697533
Total movies: 50052
Baseline Time: 2.867200 ms

Top 10 Movies:
B002YKFLBM -> 4.981818
...
```

### Optimized

```bash
nvcc -std=c++14 src/cuda_toprated_optimized.cu -o opt
./opt
```

**Output:**
```
Total reviews: 1697533
Total movies: 50052
Optimized Time: 11.794432 ms
(identical Top 10 output)
```

---

## Part 4 — Nsight Profiling

```bash
nsys profile ./baseline    # → report7.nsys-rep (Baseline: 1.4905 ms)
nsys profile ./opt         # → report8.nsys-rep (Optimized: 1.9672 ms)
```

---

## Part 5 — CUDA Sentiment Analysis

```bash
wget https://raw.githubusercontent.com/cjhutto/vaderSentiment/master/vader_lexicon.txt
nvcc src/cuda_reviewanalysis.cu -o sentiment
./sentiment
```

**Output:**
```
Lexicon loaded: 7520 words
Total reviews: 4

Sentiment Results:
POSITIVE: This movie was amazing and wonderful
NEGATIVE: This was terrible and boring
POSITIVE: Absolutely fantastic experience
NEGATIVE: Worst movie ever
```

---

## Part 6 — Sequential C++

```bash
g++ src/c_elaborate.c -o seq
./seq
time ./seq    # real: 0m0.003s
```

**Output:**
```
A1 -> Reviews: 5, Avg Length: 55.00
```

---

## Part 7 — OpenMP

```bash
g++ -fopenmp src/c_elaborate_openmp.c -o omp

OMP_NUM_THREADS=2  ./omp    # Time: ~0.221s
OMP_NUM_THREADS=4  ./omp    # Time: ~0.374s
OMP_NUM_THREADS=8  ./omp    # Time: ~0.592s
OMP_NUM_THREADS=16 ./omp    # Time: ~0.971s
```

---

## Summary of Compilation Commands

| Program | Compiler | Command |
|---|---|---|
| CUDA Baseline | nvcc | `nvcc -std=c++14 src/cuda_toprated.cu -o baseline` |
| CUDA Optimized | nvcc | `nvcc -std=c++14 src/cuda_toprated_optimized.cu -o opt` |
| CUDA Sentiment | nvcc | `nvcc src/cuda_reviewanalysis.cu -o sentiment` |
| Sequential C++ | g++ | `g++ src/c_elaborate.c -o seq` |
| OpenMP | g++ | `g++ -fopenmp src/c_elaborate_openmp.c -o omp` |
