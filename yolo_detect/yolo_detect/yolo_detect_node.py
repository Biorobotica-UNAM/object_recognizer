#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from vision_interfaces.srv import GetBbox, GetDetections
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
                ("image_topic", "/camera/rgb/image_raw"),
            ]
        )

        model_name = self.get_parameter("model_name").value
        device = self.get_parameter("device").value
        conf_th = self.get_parameter("conf_th").value
        image_topic = self.get_parameter("image_topic").value

        
        # 2. Inicialización de YOLO
        self.bridge = CvBridge()
        # Mantenemos el backend pero asegúrate de que use Half Precision (FP16) si es CUDA
        self.yolo = YoloBackend(model_name=model_name, device=device, conf_th=conf_th)

        # 3. QoS para imágenes 
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            depth=1
        )

        # 4. Estado interno
        self._last_image = None
        self._latest_detections_msg = YoloDetections()
        self._results_text = "No detections yet"

        # 5. Comunicaciones
        self.create_subscription(Image, image_topic, self._img_cb, image_qos)
        self._detections_pub = self.create_publisher(YoloDetections, "detections_list", 10)
        
        self.create_service(GetBbox, "get_bbox", self._handle_get_bbox)
        ##self.create_service(Trigger, "detection", self._handle_service_request)

        self.create_service(GetDetections, "detection", self._handle_service_request)
        self.create_service(SetBool, "enable_continuous", self._handle_enable_continuous)
        self.continuous = False
        time_period = 1.0 / 10.0  # 10 Hz
        self.timer = self.create_timer(time_period, self._do_inference)
        self.timer.cancel()

        self.get_logger().info(f"YOLO OPTIMIZED: Model={model_name} en {device}")

    def _img_cb(self, msg: Image):
        self._last_image = msg

    def _do_inference(self):
        """Método central de procesamiento para evitar duplicar código"""
        if self._last_image is None:
            return False
        
        try:
            bgr = self.bridge.imgmsg_to_cv2(self._last_image, desired_encoding="bgr8")
            
            # Inferencia
            dets = self.yolo.infer(bgr)
            
            msg = YoloDetections()
            msg.header = self._last_image.header
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

            self._latest_detections_msg = msg
            self._results_text = ", ".join(text_results) if text_results else "Nothing detected"
            
            # Publicar siempre el resultado en el tópico para que otros nodos (como Seats) lo vean
            self._detections_pub.publish(msg)
            return True
        except Exception as e:
            self.get_logger().error(f"Inference Error: {e}")
            return False

    def _handle_service_request(self, req, res):
        """El servicio ahora es mucho más rápido porque puede devolver la última detección"""
        # Si no es continuo, procesamos justo ahora
        request_time = self.get_clock().now()

        max_wait = rclpy.duration.Duration(seconds=0.5)  # Esperamos hasta 0.5s por una imagen reciente
        start_wait_time = self.get_clock().now()
        fresh_image = False

        while (self.get_clock().now() - start_wait_time).nanoseconds < max_wait.nanoseconds:
            if self._last_image is None:
                continue

            img_time = rclpy.time.Time.from_msg(self._last_image.header.stamp)

            if img_time.nanoseconds > request_time.nanoseconds:
                fresh_image = True
                break
        
        if not fresh_image:
            res.success = False
            res.message = "No recent image available for inference"
            return res

        if not self.continuous:
            success = self._do_inference()
            if not success:
                res.success = False
                res.message = "Inference failed or no image"
                res.detections = []
                return res
        
        res.success = True
        res.message = self._results_text
        res.detections = self._latest_detections_msg.detections
        return res
    
    def _handle_enable_continuous(self, req: SetBool.Request, res: SetBool.Response):
        self.continuous = req.data
        if self.continuous:
            self.timer.reset()  # Reiniciar el timeri
            self.get_logger().info("Continuous detection ENABLED")
        else:
            self.timer.cancel()  # Detener el timer de inferencia periódica
            self.get_logger().info("Continuous detection DISABLED")
        res.success = True
        res.message = "Continuous mode " + ("enabled" if self.continuous else "disabled")
        return res

    def _handle_get_bbox(self, req: GetBbox.Request, res: GetBbox.Response):
        """Versión optimizada que busca en la última detección guardada"""
        # Si los datos están muy viejos (más de 2 seg), re-procesar
        if not self.continuous:
            self._do_inference()

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
        else:
            res.success = False
        return res

def main():
    rclpy.init()
    node = YoloDetectNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()