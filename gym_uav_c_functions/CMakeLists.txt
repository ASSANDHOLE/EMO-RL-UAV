cmake_minimum_required(VERSION 2.8)
project(c_functions)

set(CMAKE_CXX_STANDARD 14)

# add_executable(${PROJECT_NAME} main.cpp)
add_library(${PROJECT_NAME} SHARED main.cpp)
