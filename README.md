# 🤖 ROS 2 PACKAGE — Kinect + YOLO Detection Fusion

This repository integrates Kinect, YOLO, and a detection + fusion pipeline in ROS 2.  
It allows you to:

- Launch the Kinect camera (PointCloud)
- Run YOLO object detection
- Query 3D object poses by class using ROS 2 services

---

## ✅ Requirements

- ROS 2 Jazzy installed  
- A ROS 2 workspace built with colcon  
- All required package dependencies installed  

---

## 📥 Clone the Repository

Navigate to your workspace:

```bash
cd <your_workspace>/src  
git clone <THIS_REPOSITORY_URL>  
git checkout yolo-detector 
```

---

## 🛠️ Build
```bash
colcon build
```

---

## 🔧 Source the Environment
```bash
source install/setup.bash
```

---

## ▶️ Run the System

### 1️⃣ Launch Kinect (PointCloud)
```bash
ros2 launch kinect_ros2 pointcloud.launch.py
```

---

### 2️⃣ Launch Detection and Fusion
```bash
ros2 launch detect_fusion detect_fusion.launch
```

---

## 📡 Available Interfaces

### 🔍 YOLO Detections Topic

/yolo_detections

Enable publishing:
```bash
ros2 service call /publish_detections/enable std_srvs/srv/Trigger "{}"
```

---

### 🍎 Get Object Pose by Class
```bash
ros2 service call /detect_by_class detector_interfaces/srv/DetectByClass "class_name: 'apple'"
```

This returns detection data and estimated pose for the requested class.

---

## 🧠 Notes

- Ensure all nodes are running before calling services  
- If no detections appear, verify that YOLO is publishing correctly  

### Useful debugging commands
```bash
ros2 topic list  
ros2 service list  
```
