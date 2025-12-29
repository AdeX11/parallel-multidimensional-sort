#pragma once
#include <vector>

struct Point {
    std::vector<float> coords; // coordinates
    float dist = 0.0f;        // distance from reference

    Point() = default;
    Point(int D) : coords(D, 0.0f), dist(0.0f) {}
};
