cmake_minimum_required(VERSION 2.8.3)
project(vfh_node)
# add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  tf
  std_msgs
)

find_package(Boost REQUIRED COMPONENTS system)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)
add_library(local_vfh
  src/vfhlite.cpp
)

add_executable(vfh_node src/vfh_node.cpp)

target_link_libraries(vfh_node
  ${catkin_LIBRARIES} local_vfh
)
