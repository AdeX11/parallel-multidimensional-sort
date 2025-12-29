#pragma once
#include <vector>
#include "Point.hpp"

// GPU entry point
void run_gpu_sort(std::vector<Point>& pts, const std::vector<float>& ref);
