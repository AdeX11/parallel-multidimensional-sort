#include "Point.hpp"
#include "cpu_distance.hpp"
#include <vector>
#include <cmath>

void compute_distances_cpu(std::vector<Point>& pts, const std::vector<float>& ref) {
    const int N = pts.size();
    const int D = ref.size(); 
    
    // Pull the reference pointer out to ensure it's treated as a constant address
    const float* __restrict r_ptr = ref.data();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        float sum = 0.0f;
        const float* __restrict p_ptr = pts[i].coords.data();

        // The compiler will version this loop for small and large D automatically
        #pragma omp simd reduction(+:sum)
        for (int j = 0; j < D; j++) {
            float diff = p_ptr[j] - r_ptr[j];
            sum += diff * diff;
        }
        pts[i].dist = sum;
    }
}
