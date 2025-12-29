#include "generate_points.hpp"
#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <string>
#include <omp.h>
#include <charconv>
#include <algorithm>
#include <iomanip>
#include <ctime>
#include <cstring>

bool run_parallel_generator(long long num_points, int dims, const std::string& filename) {
    // Open with binary + trunc to explicitly overwrite existing files
    std::ofstream out(filename, std::ios::binary | std::ios::trunc);
    if (!out) {
        std::cerr << "System Error: " << std::strerror(errno) << " (" << filename << ")" << std::endl;
        return false;
    }

    // --- High-Memory Tuning ---
    // Target 32MB per write to minimize 'omp critical' lock contention
    const int bytes_per_point = dims * 12;
    const int target_write_size = 32 * 1024 * 1024;
    
    int batch_size = std::max(1, target_write_size / std::max(1, bytes_per_point));
    batch_size = std::min(batch_size, 100000); // Cap to keep per-thread memory reasonable

    std::cout << "EPYC 7V13 Optimized Config:\n"
              << " - Threads:    " << omp_get_max_threads() << "\n"
              << " - Dimensions: " << dims << "\n"
              << " - Batch Size: " << batch_size << " points per write\n"
              << "-------------------------------------------\n";

    double start_time = omp_get_wtime();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        // Seed uniquely per thread
        std::mt19937 rng(static_cast<unsigned>(std::time(nullptr)) + tid);
        std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);

        std::string thread_buffer;
        // Pre-reserve memory to avoid reallocations during loop
        thread_buffer.reserve(batch_size * dims * 13);

        #pragma omp for schedule(static)
        for (long long i = 0; i < num_points; ++i) {
            char temp_buf[32];

            for (int d = 0; d < dims; ++d) {
                float val = dist(rng);

                // Faster than sprintf/cout: std::to_chars converts float to string directly
                auto [ptr, ec] = std::to_chars(temp_buf, temp_buf + 32, val, std::chars_format::fixed, 4);
                thread_buffer.append(temp_buf, ptr - temp_buf);

                if (d < dims - 1) thread_buffer += " ";
            }
            thread_buffer += "\n";

            // Periodic flush: Check if we reached our batch size
            if ((i + 1) % batch_size == 0) {
                #pragma omp critical
                {
                    out.write(thread_buffer.data(), thread_buffer.size());
                }
                thread_buffer.clear();
            }
        }

        // --- FINAL SAFE FLUSH ---
        // Crucial: Every thread must write its remaining points after the loop
        if (!thread_buffer.empty()) {
            #pragma omp critical
            {
                out.write(thread_buffer.data(), thread_buffer.size());
            }
            thread_buffer.clear();
        }
    }

    out.close();
    
    if (out.fail()) {
        std::cerr << "Error: File write failed (Disk full or quota exceeded?)" << std::endl;
        return false;
    }

    std::cout << "\nGeneration finished in " << (omp_get_wtime() - start_time) << "s\n";
    return true;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cout << "Usage: ./gen <num_points> <dimensions> <output_file>\n";
        return 1;
    }

    try {
        long long n = std::stoll(argv[1]);
        int d = std::stoi(argv[2]);
        std::string fname = argv[3];

        if (!run_parallel_generator(n, d, fname)) {
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Invalid arguments: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
