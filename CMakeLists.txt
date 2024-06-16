cmake_minimum_required(VERSION 3.28)
project(openGL_test)

set(CMAKE_CXX_STANDARD 17)

add_executable(openGL_test main.cpp)

target_link_libraries(openGL_test glfw GL GLU)

