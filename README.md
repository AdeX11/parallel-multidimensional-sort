# Parallel k-NN Search Tool

![Language](https://img.shields.io/badge/language-C++-blue.svg)
![Backend](https://img.shields.io/badge/backend-HIP%20%7C%20OpenMP-green.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

A high-performance command-line tool designed to perform **k-Nearest
Neighbor (k-NN)** searches on massive datasets.

This application leverages parallel computing to process millions of
multidimensional points in seconds. It automatically detects your
hardware capabilities to offer both **multi-core CPU** and
**GPU-accelerated** execution modes, making it suitable for
high-throughput data science and physics simulation workflows, as well
as RAG-style retrieval tasks.

## Key Features

-   **Massive Scale:** Optimized for datasets with $10^6+$ points and
    high dimensionality.
-   **Hardware Acceleration:**
    -   **GPU Mode:** Uses AMD HIP (Bitonic Sort) to achieve massive
        throughput on data center GPUs (MI200/MI300 series).
    -   **CPU Mode:** Uses OpenMP (Merge Sort) to fully utilize all
        available cores on standard servers.
-   **Flexible Input:** Accepts standard space-separated text files
    (CSV-style) for easy integration into existing data pipelines.
-   **Robust Math:** Implements vectorized Euclidean distance
    calculations for precision and speed.

## Prerequisites

To build and use this tool, you need:

-   **Linux Environment**
-   **C++ Compiler** with OpenMP support (GCC, Clang, or HIPCC).
-   **AMD ROCm Stack** (required only if you intend to use the GPU
    backend). For Cuda, you simply need to change the makefiile

## Installation

Clone the repository and build using the provided Makefile.

``` bash
git clone https://github.com/AdeX11/parallel-multidimensional-sort.git
cd parallel-multidimensional-sort

# Build with full optimizations
make
```

## Usage

### Syntax

``` bash
./sort <input_file> <backend> <ref_point>
```

**input_file:** Path to the dataset (space-separated text file)\
**backend:** \`cpu\` or \`gpu\`\
**ref_point:** Comma-separated coordinates, for example: \`0,0,0\` or
\`1.5,2.1,0.5\`

### Quick Start Commands

#### 1. Generate Test Data

Create a synthetic dataset with \`N\` points and \`dim\` dimensions.

``` bash
# Syntax: ./generate_points <N> <dim> <output_file>
./generate_points 10000000 3 input_10M.txt
```

#### 2. Run on GPU

Recommended for very large datasets.

``` bash
./sort input_10M.txt gpu 0,0,0
```

#### 3. Run on CPU

Multi-threaded execution using OpenMP.

``` bash
./sort input_10M.txt cpu 1.5,2.0,0.5
```

#### 4. Cleanup

Remove compiled binaries and object files.

``` bash
make clean
```

## Sample Output

To eliminate I/O overhead from performance metrics, this implementation
focuses on the compute pipeline and simply prints the closest and
farthest distance points found.

Below is a sample run on an AMD MI210 GPU with 10 million points:

``` plaintext
$ ./sort /work1/browncourse/bustudent62/input10M.txt gpu 0,0,0

--- Detailed Operation Times ---
Data Loading (mmap) Time:  31778.33 ms  (31.78 s)
Alloc Time:                    0.73 ms
Host-to-Device (H2D):       1289.90 ms
Distance Kernel:             272.47 ms
Sort Kernel:                  69.88 ms
Device-to-Host (D2H):         18.41 ms
Host Preparation:           3201.02 ms
Host Reorder:                231.44 ms

--- Result Check ---
Closest Distance: 213472592.000000
Farthest Distance: 297711392.000000
```

## Benchmarking

The repository includes a Python script (\`bench.py\`) to visualize
performance scaling across:

-   Single-threaded CPU
-   Multi-threaded CPU
-   GPU execution

**Default configuration:**

-   **N = 10,000,000** points
-   **D = 768** dimensions (commonly used for BERT and similar
    embeddings)

To run the benchmarks:

``` bash
python3 bench.py
```

## Contributing

Pull requests are welcome. For major changes, such as adding CUDA
support or further optimizing host-device memory transfers, please open
an issue first to discuss the design and approach.
