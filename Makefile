CXX = hipcc
CXXFLAGS = -O3 -fopenmp -march=znver3 -I.

# Sorting App Files
SORT_SOURCES = src/main.cpp src/cpu_distance.cpp src/cpu_mergesort.cpp src/load_points.cpp src/gpu_hip.cpp
SORT_OBJS = $(SORT_SOURCES:.cpp=.o)

# Generator Tool Files
GEN_SOURCES = src/generate_points.cpp
GEN_OBJS = $(GEN_SOURCES:.cpp=.o)

all: sort generate_points

# Link Sort App (Excludes generate_points.o)
sort: $(SORT_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o sort

# Link Gen Tool (Excludes main.o and others)
generate_points: $(GEN_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o generate_points

# Standard compile rule
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f src/*.o sort generate_points
