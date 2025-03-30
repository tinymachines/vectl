cmake_minimum_required(VERSION 3.10)
project(VectorClusterStore)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find pybind11
find_package(pybind11 REQUIRED)

# Add include directories
include_directories(${CMAKE_SOURCE_DIR}/src)

# Define source files
set(VECTOR_STORE_SRCS 
    src/vector_cluster_store.cpp
    src/kmeans_clustering.cpp
)

# Main library
add_library(vector_cluster_store SHARED ${VECTOR_STORE_SRCS})
target_link_libraries(vector_cluster_store stdc++ m)

# Python bindings
pybind11_add_module(vector_cluster_store_py
    src/python_bindings.cpp
)
target_link_libraries(vector_cluster_store_py PRIVATE vector_cluster_store)

# Test executable
add_executable(test_cluster_store
    src/test_cluster_store.cpp
)
target_link_libraries(test_cluster_store vector_cluster_store stdc++ m)

# Raw device test executable
add_executable(raw_device_test
    src/raw_device_test.cpp
)
target_link_libraries(raw_device_test stdc++ m)

# Perf device test executable
add_executable(vector_store_test
    src/vector_store_test.cpp
)
target_link_libraries(vector_store_test vector_cluster_store stdc++ m)

# Installation
install(TARGETS vector_cluster_store vector_cluster_store_py
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# Add a custom clean target for convenience
add_custom_target(clean_all
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_clean.cmake
)
