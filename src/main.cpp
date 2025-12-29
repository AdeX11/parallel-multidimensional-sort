#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cstring>
#include <chrono>
#include <iomanip>
#include <omp.h> // Added for thread control

#include "Point.hpp"
#include "load_points.hpp"
#include "cpu_distance.hpp"
#include "cpu_mergesort.hpp"
#include <hip/hip_runtime.h>
#include "gpu_hip.hpp"

void print_timing(const std::string& operation, double seconds) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << operation << " Time: " << seconds * 1000.0 << " ms (" << seconds << " s)\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " datafile backend [ref_point]\n";
        return 1;
    }

    const char* path = argv[1];
    const char* backend = argv[2];

    std::vector<Point> pts;
    int D = 0;

    // --- 1. Measure Data Loading (Parallel) ---
    // This is now measured because with mmap and 128 cores,
    // it's a major part of your performance profile.
    auto t_load_start = std::chrono::high_resolution_clock::now();
    if (!load_points(path, pts, D)) {
        std::cerr << "Could not load dataset\n";
        return 1;
    }
    auto t_load_end = std::chrono::high_resolution_clock::now();
    double load_time = std::chrono::duration<double>(t_load_end - t_load_start).count();

    // Prepare reference point
    std::vector<float> ref(D, 0.0f);
    if (argc >= 4) {
        std::stringstream ss(argv[3]);
        std::string tok;
        int idx = 0;
        while (std::getline(ss, tok, ',') && idx < D) {
            ref[idx++] = std::stof(tok);
        }
    }

    if (strcmp(backend, "cpu") == 0) {
        // Display hardware info
        int max_threads = omp_get_max_threads();
        std::cout << "\n--- Running CPU Backend (" << max_threads << " threads) ---\n";
        std::cout << "N=" << pts.size() << ", D=" << D << "\n";

        // --- 2. Measure CPU Distance Calculation ---
        auto t1_start = std::chrono::high_resolution_clock::now();
        compute_distances_cpu(pts, ref);
        auto t1_end = std::chrono::high_resolution_clock::now();
        double dist_time = std::chrono::duration<double>(t1_end - t1_start).count();

        // --- 3. Measure CPU Sorting ---
        auto t2_start = std::chrono::high_resolution_clock::now();
        // Passing only pts; the internal wrapper now handles scratchpad and tasks
        mergesort_cpu(pts);
        auto t2_end = std::chrono::high_resolution_clock::now();
        double sort_time = std::chrono::duration<double>(t2_end - t2_start).count();

        // --- Print Results ---
        std::cout << "\n--- Detailed Operation Times ---\n";
        print_timing("Data Loading (mmap)", load_time);
        print_timing("Distance Calculation", dist_time);
        print_timing("Sorting (Mergesort)", sort_time);
        std::cout << "------------------------------------\n";
        print_timing("Total Pipeline Time", load_time + dist_time + sort_time);

    } else if (strcmp(backend, "gpu") == 0) {
        std::cout << "\n--- Detailed Operation Times ---\n";
        print_timing("Data Loading (mmap)", load_time);
        run_gpu_sort(pts, ref);
    }

    std::cout << "\n--- Result Check ---\n";
    if(!pts.empty()) {
        std::cout << "Closest Distance: " << pts[0].dist << "\n";
        std::cout << "Farthest Distance: " << pts.back().dist << "\n";
    }
    return 0;
}
