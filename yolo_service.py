# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Pose, TransformStamped
from vision_interfaces.srv import YoloService
import tf2_ros
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import ros2_numpy


class ClassifyServer(Node):
    def __init__(self):
        super().__init__('classification_server')

        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.last_cloud = None
        self.last_transforms = []
        self.timer_broadcast = None

        self.debug_pub = self.create_publisher(Image, "/yolo/debug_image", 10)

        self.create_subscription(PointCloud2, '/kinect/points', self.cb_points, 10)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = YOLO("yolov8x.pt")
        self.get_logger().info(f"YOLOv8 loaded on {device}")

        self.srv = self.create_service(YoloService, 'classify', self.callback)
        self.get_logger().info("classification_service ready.")

    def cb_points(self, msg: PointCloud2):
        self.last_cloud = msg

    def start_broadcast_timer_if_needed(self):
        if self.timer_broadcast is None:
            self.timer_broadcast = self.create_timer(0.1, self._broadcast_transforms)

    def _broadcast_transforms(self):
        for t in self.last_transforms:
            t.header.stamp = self.get_clock().now().to_msg()
            self.tf_broadcaster.sendTransform(t)

    def make_transform_from_pose(self, pose: Pose, child_name: str, parent_frame: str = "kinect_depth"):
        t = TransformStamped()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_name
        t.header.stamp = self.get_clock().now().to_msg()

        t.transform.translation.x = float(pose.position.x)
        t.transform.translation.y = float(pose.position.y)
        t.transform.translation.z = float(pose.position.z)
        t.transform.rotation.w = 1.0
        return t

    def callback(self, req, resp):
        self.get_logger().info("YOLO request received")

        resp.pose = Pose()
        resp.name = ""
        resp.pt_min = []
        resp.pt_max = []
        resp.confidence = 0.0
        resp.debug_images = []

        if req.image.width == 0:
            return resp

        try:
            cv_img = self.bridge.imgmsg_to_cv2(req.image, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge failed: {e}")
            return resp

        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        results = self.model(img_rgb)
        r = results[0]

        debug_img = img_rgb.copy()

        if r.boxes is None or len(r.boxes) == 0:
            self.get_logger().info("No detections at all")
            self.publish_debug(debug_img, resp)
            return resp

        found = False

        for i in range(len(r.boxes)):
            box = r.boxes[i]

            cls_id = int(box.cls[0])
            class_name = r.names[cls_id]

            found = True


            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])

            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(debug_img, f"{class_name} {conf:.2f}",
                        (x1, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            p = Pose()

            if self.last_cloud is not None:
                try:
                    pc = ros2_numpy.numpify(self.last_cloud)

                    xs = pc['x'][y1:y2, x1:x2]
                    ys = pc['y'][y1:y2, x1:x2]
                    zs = pc['z'][y1:y2, x1:x2]

                    mx = float(np.nanmean(xs))
                    my = float(np.nanmean(ys))
                    mz = float(np.nanmean(zs))

                    if not np.isnan(mx):
                        p.position.x = mx
                        p.position.y = my
                        p.position.z = mz
                        p.orientation.w = 1.0

                        tf_name = f"{class_name}"
                        tmsg = self.make_transform_from_pose(p, tf_name)
                        self.last_transforms = [tmsg]
                        self.start_broadcast_timer_if_needed()

                except Exception as e:
                    self.get_logger().warn(f"3D projection failed: {e}")

            resp.poses.append(p) 
            resp.names.append(str(class_name)) 
            resp.pt_min.extend([float(x1), float(y1)]) 
            resp.pt_max.extend([float(x2), float(y2)]) 
            resp.confidence.append(float(conf)) 
            
            debug_bgr = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR) 
            msg = self.bridge.cv2_to_imgmsg(debug_bgr, encoding="bgr8") 
            self.debug_pub.publish(msg) 
            resp.debug_images.append(msg) 
            return resp


def main():
    rclpy.init()
    node = ClassifyServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()


