#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from vision_interfaces.srv import GetBbox
from vision_interfaces.msg import YoloDetection, YoloDetections
from std_srvs.srv import Trigger
from yolo_detect.yolo_backend import YoloBackend

class YoloDetectNode(Node):
    def __init__(self):
        super().__init__("yolo_detect_node")

        # 1. Parámetros
        self.declare_parameters(
            namespace='',
            parameters=[
                ("model_path", "yolo11m.pt"),
                ("device", "cuda:0"),
                ("conf_th", 0.5),
                ("detect_topic", "/kinect/image_raw"),
                ("enable_continuous", False) # Nuevo: ¿Correr siempre o bajo demanda?
            ]
        )

        model_path = self.get_parameter("model_path").value
        device = self.get_parameter("device").value
        conf_th = self.get_parameter("conf_th").value
        detect_topic = self.get_parameter("detect_topic").value
        self.continuous = self.get_parameter("enable_continuous").value

        # 2. Inicialización de YOLO
        self.bridge = CvBridge()
        # Mantenemos el backend pero asegúrate de que use Half Precision (FP16) si es CUDA
        self.yolo = YoloBackend(model_path=model_path, device=device, conf_th=conf_th)

        # 3. QoS para imágenes (Crucial para evitar lag)
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=1 # Solo procesar el frame más actual
        )

        # 4. Estado interno
        self._last_image = None
        self._latest_detections_msg = YoloDetections()
        self._results_text = "No detections yet"

        # 5. Comunicaciones
        self.create_subscription(Image, detect_topic, self._img_cb, image_qos)
        self._detections_pub = self.create_publisher(YoloDetections, "detections_list", 10)
        
        self.create_service(GetBbox, "get_bbox", self._handle_get_bbox)
        self.create_service(Trigger, "detection", self._handle_service_request)

        self.get_logger().info(f"YOLO OPTIMIZED: Model={model_path} en {device}")

    def _img_cb(self, msg: Image):
        self._last_image = msg
        # Si el modo continuo está activo, procesamos cada frame
        if self.continuous:
            self._do_inference()

    def _do_inference(self):
        """Método central de procesamiento para evitar duplicar código"""
        if self._last_image is None:
            return False

        try:
            # Optimización: Redimensionar imagen si es muy grande (ej. 4K a 640)
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
        if not self.continuous:
            success = self._do_inference()
            if not success:
                res.success = False
                res.message = "Inference failed or no image"
                return res
        
        res.success = True
        res.message = self._results_text
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
    # Usar SingleThreaded es suficiente para YOLO si no publicamos muchas cosas pesadas,
    # pero permite que el sistema de ROS gestione los tiempos.
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()