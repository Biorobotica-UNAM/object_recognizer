from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from vision_interfaces.srv import GetBbox
from std_srvs.srv import Trigger
from yolo_detect.yolo_backend import YoloBackend

class YoloDetectNode(Node):
    def __init__(self):
        super().__init__("yolo_detect_node", namespace="yolo")

        self.declare_parameter("model_path", "yolo11m.pt")
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("conf_th", 0.5)
        self.declare_parameter("image_topic", "/image")

        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        device = self.get_parameter("device").get_parameter_value().string_value
        conf_th = self.get_parameter("conf_th").get_parameter_value().double_value
        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value

        self.bridge = CvBridge()
        self.yolo = YoloBackend(model_path=model_path, device=device, conf_th=conf_th)

        self._last_image: Optional[Image] = None

        self.publisher_enabled = False
        self.create_subscription(Image, image_topic, self._img_cb, 10)
        self._detection_pub = self.create_publisher(Image, "detections", 10)

        self.create_service(GetBbox, "get_bbox", self._handle_get_bbox)
        self.create_service(Trigger, "publish_detections/enable", self._hanldle_publish_detections)

        self.get_logger().info(f"YOLO ready. topic={image_topic} model={model_path} device={device} conf={conf_th}")

    def _img_cb(self, msg: Image):
        self._last_image = msg

    def publish_detections(self, dets):
        # Publish image with detections drawn
        if self._last_image is None:
            return
        try:
            bgr = self.bridge.imgmsg_to_cv2(self._last_image, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return
        drawn = self.yolo.draw_detections(bgr, dets)
        drawn_msg = self.bridge.cv2_to_imgmsg(drawn, encoding="bgr8")
        drawn_msg.header = self._last_image.header
        self._detection_pub.publish(drawn_msg)

    def _hanldle_publish_detections(self, req: Trigger.Request, res: Trigger.Response):
        self.publisher_enabled = True
        res.success = True
        return res


    def _handle_get_bbox(self, req: GetBbox.Request, res: GetBbox.Response):
        if self._last_image is None or not req.class_name.strip():
            res.success = False
            return res

        try:
            bgr = self.bridge.imgmsg_to_cv2(self._last_image, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            res.success = False
            return res

        dets = self.yolo.infer(bgr)
        # self.get_logger().info(f"Detections: {dets} found.")
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

def main():
    rclpy.init()
    node = YoloDetectNode()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)

            if not node.publisher_enabled:
                continue

            if node._last_image is None:
                logger = node.get_logger()
                logger.info("Waiting for image...")
                continue

            try:
                bgr = node.bridge.imgmsg_to_cv2(node._last_image, desired_encoding="bgr8")
            except Exception as e:
                node.get_logger().error(f"cv_bridge error: {e}")
                continue

            dets = node.yolo.infer(bgr)
            node.publish_detections(dets)

    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
