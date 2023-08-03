cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(SparseConvolution)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(SOURCES main.cpp src/file_input.cpp)

add_executable(SparseConvolution 
main.cpp 
src/data_handling/file_input.cpp 
src/testing/benchmark.cpp 
src/testing/test.cpp
src/convolutions/dense_convolution.cpp
src/convolutions/sparse_convolution.cpp
src/convolutions/general_sparse.cpp)
target_link_libraries(SparseConvolution "${TORCH_LIBRARIES}")
set_property(TARGET SparseConvolution PROPERTY CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")