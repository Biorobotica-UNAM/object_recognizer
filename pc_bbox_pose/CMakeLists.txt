cmake_minimum_required(VERSION 3.8)
project(pc_bbox_pose)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(vision_interfaces REQUIRED)

add_executable(pc_bbox_pose_node src/pc_bbox_pose_node.cpp)

ament_target_dependencies(pc_bbox_pose_node
  rclcpp
  sensor_msgs
  geometry_msgs
  vision_interfaces
  tf2
  tf2_ros
  tf2_geometry_msgs
)

target_link_libraries(pc_bbox_pose_node Eigen3::Eigen)

install(TARGETS pc_bbox_pose_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
