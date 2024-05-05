cmake_minimum_required(VERSION 3.27)
project(svd_demo)

set(CMAKE_CXX_STANDARD 20)

find_package(raylib REQUIRED)

add_executable(svd main.cpp)
target_link_libraries(svd raylib)
target_compile_options(svd PRIVATE "-DMODEL_FILE=\"${CMAKE_SOURCE_DIR}/stanford-bunny.obj\"" -O3 -march=native)