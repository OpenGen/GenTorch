cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(Gen VERSION 0.1)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(${PROJECT_SOURCE_DIR} ${TORCH_INCLUDE_DIRS})

add_executable(test_address test/test_address.cpp)
target_link_libraries(test_address "${TORCH_LIBRARIES}")

add_executable(test_dml test/test_dml.cpp)
target_link_libraries(test_dml "${TORCH_LIBRARIES}")