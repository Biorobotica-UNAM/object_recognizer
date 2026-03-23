cmake_minimum_required(VERSION 3.8)
project(vision_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GetDetections.srv"
  "srv/Owlv2Service.srv"
  "srv/BboxToPose.srv"
  "srv/DetectByClass.srv"
  "srv/GetBbox.srv"
  "msg/YoloDetection.msg"
  "msg/YoloDetections.msg"
  
  DEPENDENCIES sensor_msgs geometry_msgs std_msgs
)

ament_package()




