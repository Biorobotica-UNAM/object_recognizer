cmake_minimum_required(VERSION 3.8)
project(detect_fusion)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(detector_interfaces REQUIRED)

add_executable(detect_fusion_node src/detect_fusion_node.cpp)
ament_target_dependencies(detect_fusion_node rclcpp detector_interfaces)

install(TARGETS detect_fusion_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)


ament_package()
