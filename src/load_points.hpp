#pragma once
#include <vector>
#include <string>
#include "Point.hpp"

bool load_points(const std::string& filename, std::vector<Point>& points, int& D);
