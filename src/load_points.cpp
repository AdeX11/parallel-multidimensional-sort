#include "load_points.hpp"
#include <iostream>
#include <vector>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <charconv> // For fast string-to-float conversion

bool load_points(const std::string& filename, std::vector<Point>& points, int& D) {
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1) return false;

    struct stat sb;
    fstat(fd, &sb);
    size_t length = sb.st_size;

    char* addr = static_cast<char*>(mmap(NULL, length, PROT_READ, MAP_PRIVATE, fd, 0));
    if (addr == MAP_FAILED) { close(fd); return false; }

    // 1. Determine D from the first line
    char* first_line_ptr = addr;
    D = 0;
    bool in_val = false;
    while (*first_line_ptr != '\n' && first_line_ptr < addr + length) {
        if (!isspace(*first_line_ptr)) {
            if (!in_val) { D++; in_val = true; }
        } else {
            in_val = false;
        }
        first_line_ptr++;
    }

    // 2. Determine exact N (Number of lines)
    // Since lines = points, we count '\n' to be 100% accurate
    size_t N = 0;
    #pragma omp parallel for reduction(+:N)
    for (size_t i = 0; i < length; i++) {
        if (addr[i] == '\n') N++;
    }
    // Handle files that don't end in a newline
    if (length > 0 && addr[length-1] != '\n') N++;

    // 3. Resize and PRE-ALLOCATE (The fix for the Segfault)
    points.resize(N, Point(D)); 

    // 4. Parallel Parsing
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int n_threads = omp_get_num_threads();
        size_t chunk_size = length / n_threads;
        char* my_start = addr + (tid * chunk_size);
        char* my_end = (tid == n_threads - 1) ? addr + length : addr + ((tid + 1) * chunk_size);

        // Align start to beginning of line
        if (tid != 0) {
            while (my_start < my_end && *(my_start - 1) != '\n') my_start++;
        }

        // Calculate which index in the GLOBAL vector this thread starts at
        size_t global_idx = 0;
        for (char* p = addr; p < my_start; p++) {
            if (*p == '\n') global_idx++;
        }

        char* curr = my_start;
        size_t local_count = 0;
        while (curr < my_end) {
            if (global_idx + local_count >= N) break;

            char* next;
            // Directly fill the pre-allocated vector
            for (int i = 0; i < D; ++i) {
                points[global_idx + local_count].coords[i] = std::strtof(curr, &next);
                curr = next;
            }
            local_count++;
            while (curr < my_end && (*curr == '\n' || *curr == '\r' || *curr == ' ')) curr++;
        }
    }

    munmap(addr, length);
    close(fd);
    return !points.empty();
}
