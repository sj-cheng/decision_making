# CMakeLists.txt

cmake_minimum_required(VERSION 3.5)
project(bindings)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_BUILD_TYPE Debug)
add_compile_options(-O2)


find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories( ${EIGEN3_INCLUDE_DIRS} )

pybind11_add_module(bindings bindings.cpp)