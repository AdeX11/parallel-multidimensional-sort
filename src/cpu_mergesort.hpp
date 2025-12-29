#pragma once
#include <vector>
#include "Point.hpp"


void mergesort_cpu(std::vector<Point>& pts);

// The internal recursive function (optional to keep in header)
void mergesort_recursive(std::vector<Point>& pts, std::vector<Point>& scratch, int left, int right, int grain_size);
