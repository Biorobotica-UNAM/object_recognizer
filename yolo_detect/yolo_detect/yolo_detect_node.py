#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import TransformStamped
import tf2_ros
from vision_interfaces.srv import GetDetections, BboxToPose
from vision_interfaces.msg import YoloDetection, YoloDetections
from std_srvs.srv import SetBool
from yolo_detect.yolo_backend import YoloBackend

class YoloDetectNode(Node):
    def __init__(self):
        super().__init__("yolo_detect_node")

        #  Parámetros
        self.declare_parameters(
            namespace='',
            parameters=[
                ("model_name", "yolo11m.pt"),
                ("device", "cuda"),
                ("conf_th", 0.5),
                ("image_topic", "hardware/camera/image_raw"),
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
        
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self) 
        self._current_queries = []

        self._last_image = None
        self._latest_detections_msg = YoloDetections()
        self._results_text = "No detections yet"

        #  Subs, Pubs y Clientes
        image_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=1)
        self.create_subscription(Image, image_topic, self._img_cb, image_qos, callback_group=self.cb_group)

        from std_msgs.msg import Bool as BoolMsg
        self.create_subscription(BoolMsg, 'enable_continuous',
                                 self._enable_topic_cb, 10,
                                 callback_group=self.cb_group)

        self._detections_pub = self.create_publisher(YoloDetections, "detections_list", 10)

        self.bbox_client = self.create_client(
            BboxToPose, self.bbox_to_pose_service, callback_group=self.cb_group)
        
        # Servicios
        self.create_service(GetDetections, "detection", self._handle_service_request, callback_group=self.cb_group)
        self.create_service(SetBool, "enable_continuous", self._handle_enable_continuous, callback_group=self.cb_group)
        
        self.continuous = False
        time_period = 1.0 / 10.0  # 10 Hz
        self.timer = self.create_timer(time_period, self._do_inference, callback_group=self.cb_group)
        self.timer.cancel()

        self.get_logger().info(f"YOLO NODE: {model_name} en {device} escuchando {image_topic}")

    def _img_cb(self, msg: Image):
        self._last_image = msg

    def _do_inference(self):
        if self._last_image is None:
            return False
        
        try:
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

                if d.class_name in self._current_queries:
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
        except Exception as e:
            self.get_logger().error(f"Error en callback de pose: {e}")

    def _handle_service_request(self, req: GetDetections.Request, res: GetDetections.Response):
        if self._last_image is None:
            res.success = False
            res.message = "No image received yet from topic"
            return res

        # Actualizamos las queries para que _do_inference sepa a qué sacarle pose
        self._current_queries = req.queries

        try:
            if not self.continuous:
                self._do_inference()
            
            found = []
            for d in self._latest_detections_msg.detections:
                if d.class_name in req.queries:
                    found.append(d)
            
            res.detections = found
            res.success = len(found) > 0
            
            if res.success:
                res.message = f"Detected {len(found)} objects"
            else:
                res.message = f"Nothing found. YOLO sees: {self._results_text}"
        
        except Exception as e:
            res.success = False
            res.message = f"Inference error: {str(e)}"
        
        return res
    
    def _handle_enable_continuous(self, req: SetBool.Request, res: SetBool.Response):
        self._apply_enable(req.data)
        res.success = True
        res.message = f"Continuous mode {'enabled' if self.continuous else 'disabled'}"
        return res

    # [FIX] Callback del subscriber Bool — misma lógica que el servicio
    def _enable_topic_cb(self, msg):
        self._apply_enable(msg.data)

    def _apply_enable(self, enable: bool):
        """Lógica de enable/disable unificada para servicio y subscriber."""
        self.continuous = enable
        if self.continuous:
            self.timer.reset()
            self.get_logger().info("Continuous detection ENABLED (via topic)")
        else:
            self.timer.cancel()
            self._current_queries = []
            self.get_logger().info("Continuous detection DISABLED (via topic)")

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