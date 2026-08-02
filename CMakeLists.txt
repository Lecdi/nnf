cmake_minimum_required(VERSION 3.23)
project(nnf LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

option(USE_SYSTEM_INSTALLATIONS "Use system installations for some libraries where available" ON)
option(BUILD_BENCHMARKS "Fetch Google Benchmark and build benchmarks" ON)
option(BUILD_TESTS "Fetch Catch2 and build tests" ON)
option(COMPONENT_MNIST_EXAMPLE "Build the MNIST example component" ON)

include("Dependencies.cmake")

add_subdirectory("components/nnf")

if(COMPONENT_MNIST_EXAMPLE)
    add_subdirectory("components/mnist_example")
endif()
