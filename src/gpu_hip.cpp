// src/gpu_hip.cpp
#include "gpu_hip.hpp"
#include "Point.hpp"
#include <hip/hip_runtime.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <chrono>
#include <omp.h> // Included for parallel reordering

// --- Timing Helpers ---
#define START_TIMER(event) HIP_CHECK(hipEventRecord(event##Start, 0))
#define STOP_TIMER(event) HIP_CHECK(hipEventRecord(event##Stop, 0)); HIP_CHECK(hipEventSynchronize(event##Stop))
#define GET_TIME(event) { \
    float ms; \
    HIP_CHECK(hipEventElapsedTime(&ms, event##Start, event##Stop)); \
    printf(#event " Time: %.3f ms\n", ms); \
}
// ----------------------

static inline void hip_check(hipError_t e, const char* file, int line) {
    if (e != hipSuccess) {
        fprintf(stderr, "HIP error: %s at %s:%d\n", hipGetErrorString(e), file, line);
        exit(1);
    }
}
#define HIP_CHECK(x) hip_check((x), __FILE__, __LINE__)

/* ---------------- distance kernel ---------------- */
__global__ void distance_kernel(const float* coords_flat,
                                const float* ref,
                                float* out_dists,
                                int N, int D, int M) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    if (i >= N) {
        out_dists[i] = INFINITY;
        return;
    }
    const float* row = coords_flat + (size_t)i * (size_t)D;
    float s = 0.0f;
    for (int j = 0; j < D; ++j) {
        float diff = row[j] - ref[j];
        s += diff * diff;
    }
    out_dists[i] = s;
}

/* ---------------- bitonic compare-exchange kernel ---------------- */
__global__ void bitonic_step_kernel(float* keys, int* vals, int M, int k, int j) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    int ixj = i ^ j;
    if (ixj <= i) return;

    float key_i = keys[i];
    float key_j = keys[ixj];
    int val_i = vals[i];
    int val_j = vals[ixj];

    bool ascending = ((i & k) == 0);
    if (ascending) {
        if (key_i > key_j || (key_i == key_j && val_i > val_j)) {
            keys[i] = key_j; keys[ixj] = key_i;
            vals[i] = val_j; vals[ixj] = val_i;
        }
    } else {
        if (key_i < key_j || (key_i == key_j && val_i < val_j)) {
            keys[i] = key_j; keys[ixj] = key_i;
            vals[i] = val_j; vals[ixj] = val_i;
        }
    }
}

static int next_pow2(int v) {
    if (v <= 1) return 1;
    int p = 1;
    while (p < v) p <<= 1;
    return p;
}

void run_gpu_sort(std::vector<Point>& pts, const std::vector<float>& ref) {
    int N = (int)pts.size();
    if (N == 0) return;
    int D = (int)ref.size();

    auto host_prep_start = std::chrono::high_resolution_clock::now();

    size_t coords_count = (size_t)N * (size_t)D;
    std::vector<float> h_coords(coords_count);
    
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D; ++j) {
            h_coords[(size_t)i * D + j] = pts[i].coords[j];
        }
    }

    int M = next_pow2(N);
    std::vector<int> h_idx(M);
    #pragma omp parallel for
    for (int i = 0; i < M; ++i) h_idx[i] = i < N ? i : (N + i);

    auto host_prep_stop = std::chrono::high_resolution_clock::now();

    float *d_coords, *d_ref, *d_keys;
    int *d_vals;

    hipEvent_t allocStart, allocStop, h2dStart, h2dStop, distKernelStart, distKernelStop, sortKernelStart, sortKernelStop, d2hStart, d2hStop;
    HIP_CHECK(hipEventCreate(&allocStart)); HIP_CHECK(hipEventCreate(&allocStop));
    HIP_CHECK(hipEventCreate(&h2dStart));   HIP_CHECK(hipEventCreate(&h2dStop));
    HIP_CHECK(hipEventCreate(&distKernelStart)); HIP_CHECK(hipEventCreate(&distKernelStop));
    HIP_CHECK(hipEventCreate(&sortKernelStart)); HIP_CHECK(hipEventCreate(&sortKernelStop));
    HIP_CHECK(hipEventCreate(&d2hStart));   HIP_CHECK(hipEventCreate(&d2hStop));

    START_TIMER(alloc);
    HIP_CHECK(hipMalloc(&d_coords, sizeof(float) * coords_count));
    HIP_CHECK(hipMalloc(&d_ref, sizeof(float) * D));
    HIP_CHECK(hipMalloc(&d_keys, sizeof(float) * (size_t)M));
    HIP_CHECK(hipMalloc(&d_vals, sizeof(int) * (size_t)M));
    STOP_TIMER(alloc);

    START_TIMER(h2d);
    HIP_CHECK(hipMemcpy(d_coords, h_coords.data(), sizeof(float) * coords_count, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_ref, ref.data(), sizeof(float) * D, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vals, h_idx.data(), sizeof(int) * (size_t)M, hipMemcpyHostToDevice));
    STOP_TIMER(h2d);

    int block = 256;
    int grid = (M + block - 1) / block;
    START_TIMER(distKernel);
    hipLaunchKernelGGL(distance_kernel, dim3(grid), dim3(block), 0, 0, d_coords, d_ref, d_keys, N, D, M);
    HIP_CHECK(hipGetLastError());
    STOP_TIMER(distKernel);

    START_TIMER(sortKernel);
    for (int k = 2; k <= M; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            hipLaunchKernelGGL(bitonic_step_kernel, dim3((M + block - 1) / block), dim3(block), 0, 0, d_keys, d_vals, M, k, j);
            HIP_CHECK(hipGetLastError());
        }
        HIP_CHECK(hipDeviceSynchronize());
    }
    STOP_TIMER(sortKernel);

    std::vector<int> sorted_idx(M);
    std::vector<float> sorted_keys(M);
    START_TIMER(d2h);
    HIP_CHECK(hipMemcpy(sorted_idx.data(), d_vals, sizeof(int) * (size_t)M, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(sorted_keys.data(), d_keys, sizeof(float) * (size_t)M, hipMemcpyDeviceToHost));
    STOP_TIMER(d2h);

    // --- 7. Host Reordering Timing (Parallelized) ---
    auto host_reorder_start = std::chrono::high_resolution_clock::now();
    std::vector<Point> temp_pts(N);
    for (int i = 0; i < N; ++i) {
      int old_idx = sorted_idx[i];
      temp_pts[i] = std::move(pts[old_idx]);
      temp_pts[i].dist = sorted_keys[i];
    }
    pts.swap(temp_pts);

    auto host_reorder_stop = std::chrono::high_resolution_clock::now();

    GET_TIME(alloc); GET_TIME(h2d); GET_TIME(distKernel); GET_TIME(sortKernel); GET_TIME(d2h);
    printf("Host Preparation Time: %.3f ms\n", std::chrono::duration_cast<std::chrono::nanoseconds>(host_prep_stop - host_prep_start).count() / 1000000.0f);
    printf("Host Reorder Time: %.3f ms\n", std::chrono::duration_cast<std::chrono::nanoseconds>(host_reorder_stop - host_reorder_start).count() / 1000000.0f);

    HIP_CHECK(hipEventDestroy(allocStart)); HIP_CHECK(hipEventDestroy(allocStop));
    HIP_CHECK(hipEventDestroy(h2dStart)); HIP_CHECK(hipEventDestroy(h2dStop));
    HIP_CHECK(hipEventDestroy(distKernelStart)); HIP_CHECK(hipEventDestroy(distKernelStop));
    HIP_CHECK(hipEventDestroy(sortKernelStart)); HIP_CHECK(hipEventDestroy(sortKernelStop));
    HIP_CHECK(hipEventDestroy(d2hStart)); HIP_CHECK(hipEventDestroy(d2hStop));
    HIP_CHECK(hipFree(d_coords)); HIP_CHECK(hipFree(d_ref)); HIP_CHECK(hipFree(d_keys)); HIP_CHECK(hipFree(d_vals));
}
