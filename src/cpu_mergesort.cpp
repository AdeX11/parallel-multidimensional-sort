#include <iostream>
#include <vector>
#include <omp.h>
#include <algorithm>
#include "Point.hpp"

// 1. Optimized Merge: Uses a pre-allocated buffer to avoid malloc thrashing
void merge_optimized(std::vector<Point>& pts, std::vector<Point>& scratch, int left, int mid, int right) {
    int i = left, j = mid + 1, k = left;

    while (i <= mid && j <= right) {
        if (pts[i].dist <= pts[j].dist) scratch[k++] = pts[i++];
        else scratch[k++] = pts[j++];
    }
    while (i <= mid) scratch[k++] = pts[i++];
    while (j <= right) scratch[k++] = pts[j++];

    // Copy back to original array
    for (int p = left; p <= right; ++p) {
        pts[p] = scratch[p];
    }
}

// 2. Recursive Logic: Uses tasks and a sequential cutoff for cache efficiency
void mergesort_recursive(std::vector<Point>& pts, std::vector<Point>& scratch, int left, int right, int grain_size) {
    if (left >= right) return;

    // Sequential Fallback: Below this size, the overhead of a 'task' costs more than the sort
    if (right - left < grain_size) {
        std::sort(pts.begin() + left, pts.begin() + right + 1, [](const Point& a, const Point& b) {
            return a.dist < b.dist;
        });
        return;
    }

    int mid = left + (right - left) / 2;

    #pragma omp task shared(pts, scratch)
    mergesort_recursive(pts, scratch, left, mid, grain_size);

    #pragma omp task shared(pts, scratch)
    mergesort_recursive(pts, scratch, mid + 1, right, grain_size);

    #pragma omp taskwait
    merge_optimized(pts, scratch, left, mid, right);
}

// 3. Entry Point: Sets up the parallel environment
void mergesort_cpu(std::vector<Point>& pts) {
    int n = pts.size();
    if (n <= 1) return;

    // Allocate scratchpad once (allocated on NUMA nodes based on first-touch)
    std::vector<Point> scratch(n);

    // Calculate grain size: aims for ~8 tasks per thread for load balancing
    int num_threads = omp_get_max_threads();
    int grain_size = std::max(2000, n / (num_threads * 8));

    #pragma omp parallel
    {
        // First-touch initialization for the scratchpad
        #pragma omp for
        for(int i = 0; i < n; i++) {
            scratch[i].dist = 0; 
        }

        #pragma omp single nowait
        {
            mergesort_recursive(pts, scratch, 0, n - 1, grain_size);
        }
    }
}

// Distance computation: Spreads workload across all 128 cores
void compute_distances(std::vector<Point>& pts, const std::vector<float>& ref) {
    int n = pts.size();
    int d = ref.size();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            float diff = pts[i].coords[j] - ref[j];
            sum += diff * diff;
        }
        pts[i].dist = sum;
    }
}
