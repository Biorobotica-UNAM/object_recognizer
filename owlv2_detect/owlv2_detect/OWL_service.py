# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Pose, TransformStamped
from vision_interfaces.srv import GetBbox 
from vision_interfaces.msg import YoloDetection, YoloDetections
import tf2_ros
import cv2
import numpy as np
import torch
import torchvision
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import ros2_numpy

CATEGORY = {
    "drinks": ["bottled water", "soda can", "juice box", "milk carton", "sprite", "coke"],
    "snacks": ["chips", "biscuit pack", "snack bag", "pringles", "popcorn"],
    "fruits": ["apple", "banana", "orange", "peach", "pear", "strawberry", "lemon"],
    "cuttlery": [ "plastic bowl", "fork", "knife"],
    "containers": ["trash can", "backpack", "handbag", "suitcase", "box", "basket", "dishwasher", "washing machine"],
    "electrodomestics": ["fridge", "oven", "microwave", "toaster", "kettle", "coffee maker"],
    "toys": ["lego", "doll", "action figure", "board game", "puzzle", "teddy bear"],
    "boing": ["juice box","carton box"]
}

GENERAL_CLASSES = [
    "bottled water", "soda can", "juice box", "milk carton", "coffee mug", 
    "glass cup", "paper plate", "plastic bowl", "metal spoon", "fork", 
    "knife", "apple", "banana", "orange", "cereal box", "snack bag", 
    "chips", "bread", "sponge", "soap bottle", "trash can", "backpack", 
    "laptop", "cell phone", "book", "remote control", "keys", "person", 
    "bottle", "woman", "robot", "chair", "sofa", "dining table", "shelf"
]

OBJECT_ALIASES = {
    "coffee mate": ["red bottle", "creamer bottle"],
    "sprite": ["green bottle", "soda can", "green can"],
    "pringles": ["chips can", "snack can", "potato chips can", "red can"],
    "jello": ["yellow box", "jelly", "carton box"],
    "spam": ["meat can", "spam can", "blue can"],

}

def normalize_label(detected_label: str):
    detected_label = detected_label.lower()
    for official_name, aliases in OBJECT_ALIASES.items():
        for a in aliases:
            if a in detected_label:
                return official_name
    return detected_label

class OwlServer(Node):
    def __init__(self):
        super().__init__('owl_server')


        self.declare_parameter("image_topic", "/output_camera/image_enhanced")
        self.declare_parameter("model_path", "google/owlv2-base-patch16")
        self.declare_parameter("conf_threshold", 0.28)

        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.model_path = self.get_parameter("model_path").get_parameter_value().string_value
        self.conf_th = self.get_parameter("conf_threshold").get_parameter_value().double_value
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformStaticBroadcaster(self)
        self.last_cloud = None
        self.last_image = None

        self.debug_pub = self.create_publisher(Image, "/owl/detections", 10)
        self.owl_pub = self.create_publisher(YoloDetections, "/owl/detections_list", 10)
        
        self.create_subscription(Image, self.image_topic, self.cb_image, 10)
        self.create_subscription(PointCloud2, "/kinect/points", self.cb_points, 10)

        self.get_logger().info(f"Loading OWLv2: {self.model_path}...")
        
        self.processor = Owlv2Processor.from_pretrained(self.model_path)
        self.model = Owlv2ForObjectDetection.from_pretrained(
            self.model_path
        ).to(self.device).half().eval()
        
        self.srv = self.create_service(GetBbox, "/owl/Owlv2Service", self.callback)
        
        self.get_logger().info(
            f"OWLv2 ready. topic={self.image_topic} "
            f"model={self.model_path} device={self.device} conf={self.conf_th}"
        )

    def cb_image(self, msg):
        self.last_image = msg

    def cb_points(self, msg):
        self.last_cloud = msg

    def compute_3d_from_bbox(self, pc, x1, y1, x2, y2):
        try:
            h, w = pc.shape[:2]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            r = 7 # Radio de muestreo
            x_min, x_max = max(cx-r, 0), min(cx+r, w-1)
            y_min, y_max = max(cy-r, 0), min(cy+r, h-1)

            roi = pc[y_min:y_max, x_min:x_max]
            xs = roi['x'].flatten()
            ys = roi['y'].flatten()
            zs = roi['z'].flatten()

            mask = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs) & (zs > 0.1)
            if np.count_nonzero(mask) < 5: return None

            return float(np.median(xs[mask])), float(np.median(ys[mask])), float(np.median(zs[mask]))
        except Exception as e:
            self.get_logger().warn(f"3D error: {e}")
            return None
    
    def callback(self, req, resp):
        query = req.class_name.strip().lower()
        
        # 1. Definir qué vamos a buscar
        # Si no pides nada, usamos prompts genéricos para detectar "cualquier cosa"
        if query in ["", "all", "everything", "objects"]:
            search_queries = GENERAL_CLASSES
            is_general_view = True
            self.get_logger().info("Searching for any known object")

        elif query in CATEGORY:
            search_queries = CATEGORY[query]
            is_general_view = True
            self.get_logger().info(f"Searching for category '{query}'")

        else:
            search_queries = [query]
            is_general_view = False
            self.get_logger().info(f"Searching for '{query}'")
            
        if self.last_image is None:
            self.get_logger().error("No image received yet!")
            resp.success = False
            return resp

        # Preparar imagen
        cv_img = self.bridge.imgmsg_to_cv2(self.last_image, "bgr8")
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        # OWLv2 acepta una lista de textos
        inputs = self.processor(text=[search_queries], images=img_rgb, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([img_rgb.shape[:2]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, threshold=self.conf_th, target_sizes=target_sizes
        )[0]

        if len(results["boxes"]) == 0:
            self.get_logger().info("Nothing found.")
            resp.success = False
            return resp

        # 2. Limpieza de detecciones (NMS)
        # Esto evita que salgan 20 cajas sobre el mismo objeto en modo general
        boxes = results["boxes"]
        scores = results["scores"]
        labels = results["labels"]
        
        # iou_threshold=0.3 significa que si dos cajas se enciman más del 30%, borra la de menor score
        keep = torchvision.ops.nms(boxes, scores, iou_threshold=0.5)

        detections_msg = YoloDetections()
        detections_msg.header = self.last_image.header
        
        # Variables para la respuesta del servicio
        best_match_idx = -1
        highest_score = -1.0

        # 3. Dibujar y procesar cada detección
        for idx in keep:
            box = boxes[idx].int().cpu().numpy()
            score = scores[idx].cpu().item()
            label_idx = labels[idx].item()
            
            x1, y1, x2, y2 = box
            label_name = search_queries[label_idx]
            label_name = normalize_label(label_name)
            

            detection = YoloDetection()
            detection.class_name = str(label_name)
            detection.confidence = float(score)
            detection.x_min = float(box[0])
            detection.y_min = float(box[1])
            detection.x_max = float(box[2])
            detection.y_max = float(box[3])
            detections_msg.detections.append(detection)
            
            # Color: Azul para búsqueda específica, Verde para general
            color = (255, 0, 0) if not is_general_view else (0, 255, 0)
            
            cv2.rectangle(cv_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(cv_img, f"{label_name} {score:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Guardamos el mejor resultado para el mensaje de respuesta
            if score > highest_score:
                highest_score = score
                best_match_idx = idx

        self.owl_pub.publish(detections_msg)
        self.get_logger().info(f"Detections: {[f'{d.class_name} ({d.confidence:.2f})' for d in detections_msg.detections]}")

        # 4. Responder al servicio y enviar TF (solo del mejor objeto encontrado)
        if best_match_idx != -1:
            best_box = boxes[best_match_idx].int().cpu().numpy()
            bx1, by1, bx2, by2 = best_box
            
            resp.x1, resp.y1, resp.x2, resp.y2 = int(bx1), int(by1), int(bx2), int(by2)
            resp.confidence = float(highest_score)
            resp.success = True

            # Publicar TF si hay nube de puntos
            if self.last_cloud is not None:
                pc = ros2_numpy.numpify(self.last_cloud)
                res_3d = self.compute_3d_from_bbox(pc, bx1, by1, bx2, by2)
                if res_3d:
                    mx, my, mz = res_3d
                    t = TransformStamped()
                    t.header.stamp = self.get_clock().now().to_msg()
                    t.header.frame_id = self.last_cloud.header.frame_id
                    t.child_frame_id = f"detected_{query if not is_general_view else 'object'}"
                    t.transform.translation.x = mx
                    t.transform.translation.y = my
                    t.transform.translation.z = mz
                    t.transform.rotation.w = 1.0
                    self.tf_broadcaster.sendTransform(t)

        # Publicar imagen de debug con todas las cajas visibles
        debug_msg = self.bridge.cv2_to_imgmsg(cv_img, "bgr8")
        self.debug_pub.publish(debug_msg)

        return resp


    """def callback(self, req, resp):
        query = req.class_name
        self.get_logger().info(f"Searching for: {query}")

        if self.last_image is None:
            self.get_logger().error("No image received yet!")
            resp.success = False
            return resp

        cv_img = self.bridge.imgmsg_to_cv2(self.last_image, "bgr8")
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        texts = [[query]]
        inputs = self.processor(text=texts, images=img_rgb, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([img_rgb.shape[:2]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, threshold=self.conf_th, target_sizes=target_sizes
        )[0]

        # Filtrar la mejor detección
        if len(results["boxes"]) == 0:
            self.get_logger().info(f"Nothing found for {query}")
            resp.success = False
            return resp

        best_idx = torch.argmax(results["scores"]).cpu().item()
        score = results["scores"][best_idx].cpu().item()
        box = results["boxes"][best_idx].int().cpu().numpy()

        if score < 0.35: #umbral
            resp.success = False
            return resp

        x1, y1, x2, y2 = box
        resp.x1 = int(x1)
        resp.y1 = int(y1)
        resp.x2 = int(x2)
        resp.y2 = int(y2)
        resp.confidence = float(score)
        resp.success = True

        cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(cv_img, f"{query} {score:.2f}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
        debug_msg = self.bridge.cv2_to_imgmsg(cv_img, "bgr8")
        self.debug_pub.publish(debug_msg)

        if self.last_cloud is not None:
            pc = ros2_numpy.numpify(self.last_cloud)
            res_3d = self.compute_3d_from_bbox(pc, x1, y1, x2, y2)
            if res_3d:
                mx, my, mz = res_3d
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = self.last_cloud.header.frame_id
                t.child_frame_id = f"detected_{query}"
                t.transform.translation.x = mx
                t.transform.translation.y = my
                t.transform.translation.z = mz
                t.transform.rotation.w = 1.0
                self.tf_broadcaster.sendTransform(t)

        return resp"""

def main():
    rclpy.init()
    node = OwlServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()


