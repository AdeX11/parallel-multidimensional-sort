#ifndef GENERATE_POINTS_HPP
#define GENERATE_POINTS_HPP

#include <string>

/**
 * Generates random point data in parallel.
 * @param num_points Total number of points to generate.
 * @param dims Number of dimensions per point.
 * @param filename Output path for the text file.
 * @return true if successful, false otherwise.
 */
bool run_parallel_generator(long long num_points, int dims, const std::string& filename);

#endif
