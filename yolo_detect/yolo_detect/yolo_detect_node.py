from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from vision_interfaces.srv import GetBbox
from vision_interfaces.msg import YoloDetection, YoloDetections
from std_srvs.srv import Trigger
from yolo_detect.yolo_backend import YoloBackend

class YoloDetectNode(Node):
    def __init__(self):
        super().__init__("yolo_detect_node", namespace="yolo")

        self.declare_parameter("model_path", "yolo11m.pt")
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("conf_th", 0.5)
        self.declare_parameter("detect_topic", "/kinect/image_raw")

        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        device = self.get_parameter("device").get_parameter_value().string_value
        conf_th = self.get_parameter("conf_th").get_parameter_value().double_value
        detect_topic = self.get_parameter("detect_topic").get_parameter_value().string_value

        self.bridge = CvBridge()
        self.yolo = YoloBackend(model_path=model_path, device=device, conf_th=conf_th)

        self._last_image: Optional[Image] = None
        self.create_subscription(Image, detect_topic, self._img_cb, 10)
        self._detections_pub = self.create_publisher(YoloDetections, "detections_list", 10)
        # Servicios
        self.create_service(GetBbox, "get_bbox", self._handle_get_bbox)
        self.create_service(Trigger, "Detection", self._handle_publish_detections)

        self.get_logger().info(f"YOLO ready. topic={detect_topic} model={model_path} device={device} conf={conf_th}")

    def _img_cb(self, msg: Image):
        self._last_image = msg

    def _handle_publish_detections(self, req: Trigger.Request, res: Trigger.Response):
        if self._last_image is None:
            res.success = False
            res.message = "No image received yet"
            return res

        try:
            bgr = self.bridge.imgmsg_to_cv2(self._last_image, desired_encoding="bgr8")
            dets = self.yolo.infer(bgr)
            detections_msg = YoloDetections()
            detections_msg.header = self._last_image.header
            
            results_string_list = []
            
            for d in dets:
                x1, y1, x2, y2 = d.xyxy
                detection = YoloDetection()
                detection.class_name = d.class_name
                detection.confidence = float(d.conf)
                detection.x_min = float(x1)
                detection.y_min = float(y1)
                detection.x_max = float(x2)
                detection.y_max = float(y2)
                
                # Agregamos a la lista del mensaje
                detections_msg.detections.append(detection)
                det_info = (
                    f"{{class: {d.class_name}, "
                    f"conf: {d.conf:.2f}, "
                    f"bbox: [X_min: {x1:.2f}, Y_min: {y1:.2f}, X_max: {x2:.2f}, Y_max: {y2:.2f}]}}"
                )
                results_string_list.append(det_info)
            
            self._detections_pub.publish(detections_msg)
            
            res.success = True
            res.message = "\n".join(results_string_list)
            
            self.get_logger().info(f"Inference: {len(dets)} objects detected.")
            #self.get_logger().info(f"Detections details:\n{res.message}")
            
        except Exception as e:
            self.get_logger().error(f"ERROR: {e}")
            res.success = False
            res.message = f"Error: {str(e)}"
            
        return res

    def _handle_get_bbox(self, req: GetBbox.Request, res: GetBbox.Response):
        if self._last_image is None or not req.class_name.strip():
            res.success = False
            return res

        try:
            bgr = self.bridge.imgmsg_to_cv2(self._last_image, desired_encoding="bgr8")
            dets = self.yolo.infer(bgr)
            best = self.yolo.pick_best_by_class(dets, req.class_name)
            
            if best is None:
                res.success = False
                return res

            x1, y1, x2, y2 = best.xyxy
            res.success = True
            res.header = self._last_image.header
            res.x1, res.y1, res.x2, res.y2 = int(x1), int(y1), int(x2), int(y2)
            res.confidence = float(best.conf)
            res.detected_class = best.class_name
            return res
        except Exception as e:
            self.get_logger().error(f"Error en GetBbox: {e}")
            res.success = False
            return res

def main():
    rclpy.init()
    node = YoloDetectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()