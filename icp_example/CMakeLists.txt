cmake_minimum_required(VERSION 2.8.3)
project(icp_example)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

#find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
#find_package(Eigen REQUIRED)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${eigen_INCLUDE_DIRS})


add_executable(icp_exmaple src/icp_example.cpp)
target_link_libraries(icp_exmaple ${PCL_LIBRARIES})

#${eigen_LIBRARIES}


