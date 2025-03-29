cmake_minimum_required(VERSION 3.10)
project(VectorClusterStore)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find pybind11
find_package(pybind11 REQUIRED)

# Add include directories
include_directories(${CMAKE_SOURCE_DIR})

# Main library
add_library(vector_cluster_store SHARED
    vector_cluster_store.cpp
    kmeans_clustering.cpp
)
target_link_libraries(vector_cluster_store stdc++ m)

# Python bindings
pybind11_add_module(vector_cluster_store_py
    python_bindings.cpp
)
target_link_libraries(vector_cluster_store_py PRIVATE vector_cluster_store)

# Test executable
add_executable(test_cluster_store
    test_cluster_store.cpp
)
target_link_libraries(test_cluster_store vector_cluster_store stdc++ m)

# Raw device test executable
add_executable(raw_device_test
    raw_device_test.cpp
)
target_link_libraries(raw_device_test stdc++ m)

# Installation
install(TARGETS vector_cluster_store vector_cluster_store_py
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
