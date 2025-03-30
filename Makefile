CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2 -Isrc
LDFLAGS = -pthread

# Source files
VECTOR_STORE_SRCS = src/vector_cluster_store.cpp src/kmeans_clustering.cpp
TEST_STORE_SRCS = src/test_cluster_store.cpp
RAW_DEVICE_SRCS = src/raw_device_test.cpp
PERF_TEST_SRCS = src/vector_store_test.cpp

# Object files
VECTOR_STORE_OBJS = $(VECTOR_STORE_SRCS:.cpp=.o)
TEST_STORE_OBJS = $(TEST_STORE_SRCS:.cpp=.o)
RAW_DEVICE_OBJS = $(RAW_DEVICE_SRCS:.cpp=.o)
PERF_TEST_OBJS = $(PERF_TEST_SRCS:.cpp=.o)

# Header files
HEADERS = src/clustering_interface.h src/kmeans_clustering.h src/vector_cluster_store.h src/logger.h

# Targets
.PHONY: all clean

# Test executables
all: test_cluster_store raw_device_test vector_store_test

# Library target (not compiled separately but included in executables)
libvector_store: $(VECTOR_STORE_OBJS)

# Main executables
test_cluster_store: $(VECTOR_STORE_OBJS) $(TEST_STORE_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) -lm

raw_device_test: $(RAW_DEVICE_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Fix: Include VECTOR_STORE_OBJS in the linking step
vector_store_test: $(VECTOR_STORE_OBJS) $(PERF_TEST_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) -lm

# Compile rule for .cpp files
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f src/*.o test_cluster_store raw_device_test vector_store_test