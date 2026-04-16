# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Pose, TransformStamped
from vision_interfaces.srv import GetBbox, BboxToPose
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
<<<<<<< HEAD
        self.declare_parameter("conf_threshold", 0.25)
        self.declare_parameter("bbox_to_pose_service", "bbox_to_pose")
=======
        self.declare_parameter("conf_threshold", 0.28)
>>>>>>> 9b45306698ab320b7a325ffd217c3c2698f2abfc

        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.model_path = self.get_parameter("model_path").get_parameter_value().string_value
        self.conf_th = self.get_parameter("conf_threshold").get_parameter_value().double_value
        self.bbox_to_pose_service = self.get_parameter("bbox_to_pose_service").get_parameter_value().double_value
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.bridge = CvBridge()
<<<<<<< HEAD
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
=======
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.last_cloud = None
>>>>>>> 9b45306698ab320b7a325ffd217c3c2698f2abfc
        self.last_image = None

        self.debug_pub = self.create_publisher(Image, "/owl/detections", 10)
        self.owl_pub = self.create_publisher(YoloDetections, "/owl/detections_list", 10)
        
        self.create_subscription(Image, self.image_topic, self.cb_image, 10)
<<<<<<< HEAD
=======
        self.create_subscription(
            PointCloud2,
            "/hardware/camera/points",
            self.cb_points,
            rclpy.qos.qos_profile_sensor_data
        )
>>>>>>> 9b45306698ab320b7a325ffd217c3c2698f2abfc

        self.bbox_client = self.create_client(BboxToPose, self.bbox_to_pose_service)
        while not self.bbox_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'waiting for service {self.bbox_to_pose_service}...')

        self.get_logger().info(f"Loading OWLv2: {self.model_path}...")    
        self.processor = Owlv2Processor.from_pretrained(self.model_path)
        self.model = Owlv2ForObjectDetection.from_pretrained(
            self.model_path
        ).to(self.device).half().eval()
        
        self.srv = self.create_service(GetBbox, "/owl/Owlv2Service", self.callback)
        
        self.get_logger().info(
            f"OWLv2 ready. topic={self.image_topic} "
            f"service={self.bbox_to_pose_service} "
            f"model={self.model_path} device={self.device} conf={self.conf_th}"
        )

    def cb_image(self, msg):
        self.last_image = msg
<<<<<<< HEAD
    
=======

    def cb_points(self, msg):
        self.last_cloud = msg

    def compute_3d_from_bbox(self, pc, x1, y1, x2, y2):
        try:
            if isinstance(pc, dict):
                if 'xyz' in pc:
                    # El formato que tienes es un array (H, W, 3)
                    xyz_map = pc['xyz']
                    x_map = xyz_map[:, :, 0]
                    y_map = xyz_map[:, :, 1]
                    z_map = xyz_map[:, :, 2]
                elif 'x' in pc:
                    x_map, y_map, z_map = pc['x'], pc['y'], pc['z']
                else:
                    self.get_logger().error(f"Keys encontradas: {pc.keys()}. No se reconoce formato.")
                    return None
            else:
                x_map, y_map, z_map = pc['x'], pc['y'], pc['z']

            h, w = z_map.shape[:2]

            # Centro del BBox
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # ROI de 10x10 píxeles alrededor del centro para mayor estabilidad
            r = 5
            y_idx, x_idx = np.ogrid[max(cy-r, 0):min(cy+r, h), 
                                    max(cx-r, 0):min(cx+r, w)]

            xs = x_map[y_idx, x_idx].flatten()
            ys = y_map[y_idx, x_idx].flatten()
            zs = z_map[y_idx, x_idx].flatten()

            # Filtrar NaNs, ceros e infinitos
            mask = np.isfinite(zs) & (zs > 0.1) & (zs < 10.0)
            
            if np.count_nonzero(mask) < 5:
                return None

            # Mediana para evitar ruidos de bordes
            return (
                float(np.median(xs[mask])),
                float(np.median(ys[mask])),
                float(np.median(zs[mask]))
            )

        except Exception as e:
            self.get_logger().error(f"Error en compute_3d: {e}")
            return None
>>>>>>> 9b45306698ab320b7a325ffd217c3c2698f2abfc
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

        keep = torchvision.ops.nms(results["boxes"], results["scores"], iou_threshold=0.5)

        # Procesar la mejor detección para el servicio
        best_idx = keep[0]
        best_box = results["boxes"][best_idx].int().cpu().numpy()
        best_score = results["scores"][best_idx].cpu().item()

        # Rellenar respuesta básica
        resp.x1, resp.y1, resp.x2, resp.y2 = map(int, best_box)
        resp.confidence = float(best_score)

        pose_req = BboxToPose.Request()
        pose_req.x1 = resp.x1
        pose_req.y1 = resp.y1
        pose_req.x2 = resp.x2
        pose_req.y2 = resp.y2

        # Usamos call_async y esperamos el resultado de forma bloqueante 
        future = self.bbox_client.call_async(pose_req)
        # sin bloquear todo el sistema:
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

        if future.done():
            pose_res = future.result()
            if pose_res.success:
                self.get_logger().info(f"3D Pose received...")

                # Publicar TF
                t = TransformStamped()
                t.header = pose_res.pose.header
                t.child_frame_id = f"detected_{query[:10]}"
                t.transform.translation.x = pose_res.pose.position.x
                t.transform.translation.y = pose_res.pose.position.y
                t.transform.translation.z = pose_res.pose.position.z
                t.transform.rotation = pose_res.pose.orientation
                self.tf_broadcaster.sendTransform(t)
                resp.success = True
            else:
                self.get_logger().warn("C++ node failed to compute 3D pose")
                resp.success = False
        else:
            self.get_logger().error("BboxToPose service timeout")
            resp.success = False

<<<<<<< HEAD
        # Debug visualization
        cv2.rectangle(cv_img, (resp.x1, resp.y1), (resp.x2, resp.y2), (0, 255, 0), 2)
=======
            # Publicar TF si hay nube de puntos
            if self.last_cloud is None:
                self.get_logger().warn("No point cloud received yet, TF not published")
            else:
                pc = ros2_numpy.numpify(self.last_cloud)

                self.get_logger().info(f"pc type: {type(pc)}")
                if isinstance(pc, dict):
                    self.get_logger().info(f"pc keys: {list(pc.keys())}")
                else:
                    self.get_logger().info(f"pc dtype: {pc.dtype}")
                    if hasattr(pc, "dtype") and pc.dtype.names is not None:
                        self.get_logger().info(f"pc fields: {pc.dtype.names}")
                res_3d = self.compute_3d_from_bbox(pc, bx1, by1, bx2, by2)

                if res_3d is None:
                    self.get_logger().warn("No valid 3D point from bbox, TF not published")
                else:
                    mx, my, mz = res_3d

                    safe_query = (query if not is_general_view else "object").strip().lower().replace(" ", "_")

                    t = TransformStamped()
                    t.header.stamp = self.get_clock().now().to_msg()
                    t.header.frame_id = self.last_cloud.header.frame_id
                    t.child_frame_id = f"detected_{safe_query}"
                    t.transform.translation.x = mx
                    t.transform.translation.y = my
                    t.transform.translation.z = mz
                    t.transform.rotation.x = 0.0
                    t.transform.rotation.y = 0.0
                    t.transform.rotation.z = 0.0
                    t.transform.rotation.w = 1.0

                    self.tf_broadcaster.sendTransform(t)

                    self.get_logger().info(
                        f"Published static TF {t.child_frame_id} wrt {t.header.frame_id}: "
                        f"x={mx:.3f}, y={my:.3f}, z={mz:.3f}"
                    )

        # Publicar imagen de debug con todas las cajas visibles
>>>>>>> 9b45306698ab320b7a325ffd217c3c2698f2abfc
        debug_msg = self.bridge.cv2_to_imgmsg(cv_img, "bgr8")
        self.debug_pub.publish(debug_msg)

        return resp

def main():
    rclpy.init()
    node = OwlServer()
    executor = rclpy.executors.MultiThreadedExecutor()
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




