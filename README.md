# CUDA & OpenMP Accelerated Amazon Review Analytics

**Course:** CS422 — Parallel Computing | BITS Pilani, Pilani Campus
**Author:** Gaurang Gupta (2023A7PS0648P)
**Hardware:** NVIDIA GeForce RTX 3050 (4GB) · CUDA 12.9 · Ubuntu 24.04
**Stack:** CUDA C++ · OpenMP · Python · NVIDIA Nsight Systems

---

## What This Project Does

End-to-end parallel analysis of **1,697,533 Amazon Movies & TV reviews** across two compute paradigms — GPU (CUDA) and CPU (OpenMP).

| Program | Paradigm | Task |
|---|---|---|
| `cuda_toprated.cu` | CUDA baseline | Top-10 rated movies via global `atomicAdd` reduction |
| `cuda_toprated_optimized.cu` | CUDA optimized | Same, with per-block shared-memory hash table |
| `cuda_reviewanalysis.cu` | CUDA | VADER lexicon sentiment classification |
| `c_elaborate.c` | Sequential C++ | Detect reviewers with ≥5 elaborate reviews (≥50 words each) |
| `c_elaborate_openmp.c` | OpenMP | Same, parallelized across 2–16 threads |

---

## Results

### CUDA — Top-10 Movie Extraction (1,697,533 reviews · 50,052 movies)

| Kernel | Standalone (`cudaEventRecord`) | Under Nsight Profiler |
|---|---|---|
| Baseline (`atomicAdd` → global memory) | **2.8672 ms** | 1.4905 ms |
| Optimized (shared-memory hash table) | **11.7944 ms** | 1.9672 ms |
| Ratio (optimized / baseline) | **4.11× slower** | 1.32× slower |

**Key finding:** The shared-memory hash table optimization is slower on RTX 3050.
With ~34 reviews per movie on average, hash collisions inside the 512-slot
per-block table are frequent, and the initialization + collision-resolution
overhead exceeds the benefit. RTX 3050 global atomics hitting L2 cache are
already very fast. This is a documented hardware/workload-dependency effect,
not a bug. Both kernels produce **identical Top-10 output** — correctness verified.

**Top 10 Movies by Average Rating** (minimum 50 reviews):

| Rank | ASIN | Avg Rating |
|---|---|---|
| 1 | B002YKFLBM | 4.9818 |
| 2 | B003L77GCE | 4.9531 |
| 3 | B00006B1HI | 4.9531 |
| 4 | B00A27O0N4 | 4.9506 |
| 5 | B000QUEQ86 | 4.9487 |
| 6 | B007I1Q4MM | 4.9474 |
| 7 | B000RPCJB6 | 4.9423 |
| 8 | B000EMGIDC | 4.9400 |
| 9 | B006W9KNXC | 4.9338 |
| 10 | B0024OW1QQ | 4.9333 |

---

### OpenMP — Elaborate Reviewer Detection

| Configuration | Time | vs Sequential |
|---|---|---|
| Sequential (`./seq`) | **0.003s** | 1× (baseline) |
| OpenMP 2 threads | ~0.221s | **74× slower** |
| OpenMP 4 threads | ~0.374s | **125× slower** |
| OpenMP 8 threads | ~0.592s | **197× slower** |
| OpenMP 16 threads | ~0.971s | **324× slower** |

**Key finding:** Performance degrades monotonically with thread count.
The `#pragma omp critical` guard around `fgets()` serializes all file I/O —
every thread blocks waiting for a single lock to read one line.
Synchronization overhead completely dominates useful computation.

**Amdahl's Law analysis:** With serial fraction f ≈ 0.99:
```
Max speedup = 1 / (f + (1−f)/N) ≈ 1.01   (for any N)
```
This matches observations. Adding threads only adds scheduling overhead
without enabling parallel work — a textbook demonstration of why
identifying and eliminating serial bottlenecks is the prerequisite for
effective parallelization.

---

### Sentiment Analysis — VADER Lexicon (7,520 words)

| Review | Label |
|---|---|
| "This movie was amazing and wonderful" | POSITIVE |
| "This was terrible and boring" | NEGATIVE |
| "Absolutely fantastic experience" | POSITIVE |
| "Worst movie ever" | NEGATIVE |

---

## Quick Start

### Prerequisites

```bash
nvcc --version      # CUDA Toolkit 11+ required
nvidia-smi          # GPU must be visible
g++ --version       # GCC with OpenMP support
python3 --version   # Python 3.x
```

### Step 1 — Get the dataset

```bash
wget https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz
gunzip reviews_Movies_and_TV_5.json.gz
python3 src/preprocess.py          # generates reviews.csv
```

### Step 2 — Get VADER lexicon

```bash
wget https://raw.githubusercontent.com/cjhutto/vaderSentiment/master/vader_lexicon.txt
```

### Step 3 — Compile and run everything

```bash
cd scripts
bash run_all.sh
```

### Step 4 — Run benchmarks

```bash
cd scripts
bash benchmark.sh
```

---

## Manual Compilation

```bash
# CUDA baseline
nvcc -std=c++14 src/cuda_toprated.cu -o baseline

# CUDA optimized
nvcc -std=c++14 src/cuda_toprated_optimized.cu -o opt

# CUDA sentiment analysis
nvcc src/cuda_reviewanalysis.cu -o sentiment

# Sequential C++
g++ src/c_elaborate.c -o seq

# OpenMP
g++ -fopenmp src/c_elaborate_openmp.c -o omp
```

---

## CUDA Optimization Details

### Baseline (`cuda_toprated.cu`)
- Each thread processes one review record
- Direct `atomicAdd` to global memory arrays (`sum[]`, `count[]`)
- One kernel launch, simple and fast on modern GPUs with L2 cache

### Optimized (`cuda_toprated_optimized.cu`)
- Per-block shared-memory open-addressing hash table (512 slots)
- Partial sums accumulated in shared memory; one flush to global per block
- Designed to reduce global atomic pressure on high-collision workloads
- **Observed slower on RTX 3050** due to collision overhead at ~34 reviews/movie

### Sentiment Analysis (`cuda_reviewanalysis.cu`)
- Each thread processes one review end-to-end
- VADER lexicon (7,520 entries) loaded into GPU global memory
- Per-thread word tokenization and lexicon lookup in device code
- Score > 0 → POSITIVE; Score < 0 → NEGATIVE

---

## Repository Structure

```
cuda-openmp-review-analytics/
│
├── README.md                        ← you are here
├── .gitignore
├── LICENSE
│
├── src/
│   ├── cuda_toprated.cu             ← Part (a): CUDA baseline
│   ├── cuda_toprated_optimized.cu   ← Part (b): CUDA optimized
│   ├── cuda_reviewanalysis.cu       ← Part (d): CUDA sentiment
│   ├── c_elaborate.c                ← Part (e): Sequential C++
│   ├── c_elaborate_openmp.c         ← Part (f): OpenMP parallel
│   └── preprocess.py                ← Dataset preprocessing
│
├── docs/
│   ├── Design_Document.pdf          ← Full report with analysis
│   ├── Implementation_Guide.md      ← All commands reference
│   └── screenshots/                 ← Terminal output screenshots
│
├── data/
│   ├── README.md                    ← Dataset download instructions
│   ├── reviews_sentiment.csv        ← Sample: 4 reviews for sentiment
│   └── reviews_text.csv             ← Sample: 5 reviews for elaborate test
│
├── results/
│   ├── top10_movies.txt             ← Actual program output
│   ├── sentiment_output.txt         ← Sentiment results
│   ├── cuda_speedup.txt             ← CUDA timing analysis
│   └── openmp_scaling.txt           ← OpenMP thread scaling analysis
│
└── scripts/
    ├── run_all.sh                   ← Compile + run all programs
    └── benchmark.sh                 ← Timing benchmarks
```

---

## Requirements

| Component | Version |
|---|---|
| NVIDIA GPU | Compute Capability ≥ 6.0 |
| CUDA Toolkit | 11+ (tested on 12.9) |
| GCC | Any version with `-fopenmp` |
| Python | 3.x (standard library only) |
| OS | Linux (tested on Ubuntu 24.04) |

---

## Design Document

Full performance analysis, optimization rationale, hardware discussion,
and Amdahl's Law derivation:
[`docs/Design_Document.pdf`](docs/Design_Document.pdf)
