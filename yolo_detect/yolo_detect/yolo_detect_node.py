#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import TransformStamped
import torch
import tf2_ros
import time
from vision_interfaces.srv import GetBbox, GetDetections, BboxToPose
from vision_interfaces.msg import YoloDetection, YoloDetections
from std_srvs.srv import Trigger, SetBool
from yolo_detect.yolo_backend import YoloBackend

class YoloDetectNode(Node):
    def __init__(self):
        super().__init__("yolo_detect_node")

        # 1. Parámetros
        self.declare_parameters(
            namespace='',
            parameters=[
                ("model_name", "yolo11m.pt"),
                ("device", "cuda"),
                ("conf_th", 0.5),
                ("image_topic", "/hardware/camera/image_raw"),
            ]
        )

        model_name = self.get_parameter("model_name").value
        device = self.get_parameter("device").value
        conf_th = self.get_parameter("conf_th").value
        image_topic = self.get_parameter("image_topic").value
        self.declare_parameter("bbox_to_pose_service", "bbox_to_pose")
        
        # 2. Inicialización de YOLO
        self.bridge = CvBridge()
        self.yolo = YoloBackend(model_name=model_name, device=device, conf_th=conf_th)
        self.bbox_to_pose_service = self.get_parameter("bbox_to_pose_service").get_parameter_value().string_value

        self.cb_group = ReentrantCallbackGroup()
        
        # --- CORRECCIÓN 1: Broadcaster Dinámico y Target Class inicializado ---
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.target_class = None 
        # ---------------------------------------------------------------------

        self._last_image = None
        self._latest_detections_msg = YoloDetections()
        self._results_text = "No detections yet"

        # 3. QoS y Comunicaciones
        image_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=1)
        self.create_subscription(Image, image_topic, self._img_cb, image_qos, callback_group=self.cb_group)
        self._detections_pub = self.create_publisher(YoloDetections, "detections_list", 10)

        self.bbox_client = self.create_client(
            BboxToPose, self.bbox_to_pose_service, callback_group=self.cb_group)
        
        self.create_service(GetBbox, "get_bbox", self._handle_get_bbox, callback_group=self.cb_group)
        self.create_service(GetDetections, "detection", self._handle_service_request, callback_group=self.cb_group)
        self.create_service(SetBool, "enable_continuous", self._handle_enable_continuous, callback_group=self.cb_group)
        
        self.continuous = False
        time_period = 1.0 / 10.0  # 10 Hz
        self.timer = self.create_timer(time_period, self._do_inference, callback_group=self.cb_group)
        self.timer.cancel()

        self.get_logger().info(f"YOLO OPTIMIZED: Model={model_name} en {device} on {image_topic}")

    def _img_cb(self, msg: Image):
        self._last_image = msg

    def _do_inference(self):
        if self._last_image is None:
            return False
        
        try:
            # --- CORRECCIÓN 2: Asegurar el header de la imagen para el mensaje ---
            img_header = self._last_image.header
            bgr = self.bridge.imgmsg_to_cv2(self._last_image, desired_encoding="bgr8")
            
            dets = self.yolo.infer(bgr)
            
            msg = YoloDetections()
            msg.header = img_header
            text_results = []

            for d in dets:
                x1, y1, x2, y2 = d.xyxy
                det_msg = YoloDetection()
                det_msg.class_name = d.class_name
                det_msg.confidence = float(d.conf)
                det_msg.x_min, det_msg.y_min = float(x1), float(y1)
                det_msg.x_max, det_msg.y_max = float(x2), float(y2)
                msg.detections.append(det_msg)
                text_results.append(f"{d.class_name} ({d.conf:.2f})")

                # Filtro Target Class (solo pide Pose y publica TF si coincide)
                if self.target_class is not None and d.class_name == self.target_class:
                    if self.bbox_client.service_is_ready():
                        req = BboxToPose.Request()
                        req.header = img_header
                        req.x1, req.y1 = int(det_msg.x_min), int(det_msg.y_min)
                        req.x2, req.y2 = int(det_msg.x_max), int(det_msg.y_max)

                        future = self.bbox_client.call_async(req)
                        future.add_done_callback(
                            lambda f, name=d.class_name: self._pose_callback(f, name)
                        )

            self._latest_detections_msg = msg
            self._results_text = ", ".join(text_results) if text_results else "Nothing detected"
            self._detections_pub.publish(msg)
            return True

        except Exception as e:
            self.get_logger().error(f"Inference Error: {e}")
            return False

    def _pose_callback(self, future, class_name):
        try:
            response = future.result()
            if response.success:
                t = TransformStamped()
                t.header = response.pose.header
                t.child_frame_id = f"yolo_{class_name}"
                t.transform.translation.x = response.pose.pose.position.x
                t.transform.translation.y = response.pose.pose.position.y
                t.transform.translation.z = response.pose.pose.position.z
                t.transform.rotation = response.pose.pose.orientation
                self.tf_broadcaster.sendTransform(t)
            else:
                self.get_logger().warn(f"No se pudo obtener profundidad para {class_name}", throttle_duration_sec=2.0)
        except Exception as e:
            self.get_logger().error(f"Error en callback de pose: {e}")

    def _handle_service_request(self, req, res):
        request_time = self.get_clock().now()
        max_wait = rclpy.duration.Duration(seconds=0.5)
        start_wait_time = self.get_clock().now()
        fresh_image = False

        while (self.get_clock().now() - start_wait_time).nanoseconds < max_wait.nanoseconds:
            if self._last_image is None: continue
            img_time = rclpy.time.Time.from_msg(self._last_image.header.stamp)
            if img_time.nanoseconds > request_time.nanoseconds:
                fresh_image = True
                break
        
        if not fresh_image and not self.continuous:
            res.success = False
            res.message = "No recent image available"
            return res

        if not self.continuous:
            self._do_inference()
        
        res.success = True
        res.message = self._results_text
        res.detections = self._latest_detections_msg.detections
        return res
    
    def _handle_enable_continuous(self, req: SetBool.Request, res: SetBool.Response):
        self.continuous = req.data
        if self.continuous:
            self.timer.reset()
            self.get_logger().info("Continuous detection ENABLED")
        else:
            self.timer.cancel()
            self.target_class = None
            self.get_logger().info("Continuous detection DISABLED")
        res.success = True
        res.message = f"Continuous mode {'enabled' if self.continuous else 'disabled'}"
        return res

    def _handle_get_bbox(self, req: GetBbox.Request, res: GetBbox.Response):
        self.target_class = req.class_name 
        self.get_logger().info(f"Received request for class: '{req.class_name}'")
        
        if not self.continuous:
            self._do_inference()

        # --- CORRECCIÓN 3: Protección de Header para el nodo C++ ---
        if self._latest_detections_msg.header.stamp.sec == 0:
            res.header.stamp = self.get_clock().now().to_msg()
            res.header.frame_id = "kinect_link" # O tu frame_id real
        else:
            res.header = self._latest_detections_msg.header
        # -----------------------------------------------------------

        best = None
        max_conf = -1.0
        for d in self._latest_detections_msg.detections:
            if d.class_name == req.class_name and d.confidence > max_conf:
                best = d
                max_conf = d.confidence

        if best:
            res.success = True
            res.x1, res.y1 = int(best.x_min), int(best.y_min)
            res.x2, res.y2 = int(best.x_max), int(best.y_max)
            res.confidence = best.confidence
            res.detected_class = best.class_name
            self.get_logger().info(f"SUCCESS: Found '{req.class_name}'")
        else:
            res.success = False
            self.get_logger().warn(f"FAILED: '{req.class_name}' not found")
        return res

def main():
    rclpy.init()
    node = YoloDetectNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()