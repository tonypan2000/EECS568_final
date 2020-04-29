cmake_minimum_required(VERSION 2.8)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++11 -O2")

project(map)

# eigen 
include_directories("/usr/include/eigen3/")

# octomap 
find_package(octomap REQUIRED)
include_directories(${OCTOMAP_INCLUDE_DIRS})

# openCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(mapping src/buildmap.cpp)
target_link_libraries(mapping ${OpenCV_LIBS} ${OCTOMAP_LIBRARIES})
