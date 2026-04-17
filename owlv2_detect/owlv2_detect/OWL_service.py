# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from vision_interfaces.srv import GetBbox, BboxToPose
from vision_interfaces.msg import YoloDetections
import tf2_ros
import cv2
import numpy as np
import torch
import torchvision
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import time

CATEGORY = {
    "trash": ["bottled water", "soda can", "juice box", "milk carton", "sprite", "coke", "sprite can", "fanta can", "coke can", "milk bottle"],
    "snacks": ["chips", "biscuit pack", "snack bag", "pringles", "popcorn"],
    "fruits": ["apple", "banana", "orange", "peach", "pear", "strawberry", "lemon"],
    "cuttlery": ["plastic bowl", "fork", "knife"],
    "containers": ["backpack", "handbag", "suitcase", "box", "basket", "dishwasher", "washing machine"],
    "electrodomestics": ["fridge", "oven", "microwave", "toaster", "kettle", "coffee maker"],
    "toys": ["lego", "doll", "action figure", "board game", "puzzle", "teddy bear"],
    "tableware": ["red plate","mug", "cup"],
    #"boing": ["juice box", "carton box"]
}

GENERAL_CLASSES = [
    "bottled water", "soda can", "juice box", "milk carton", "coffee mug", 
    "glass cup", "paper plate", "plastic bowl", "metal spoon", "fork", 
    "knife", "apple", "banana", "orange", "cereal box", "snack bag", 
    "chips", "bread", "sponge", "soap bottle", "trash can", "backpack", 
    "laptop", "cell phone", "book", "remote control", "keys", "person", 
    "bottle", "woman", "robot", "chair", "sofa", "dining table", "shelf"
]

class OwlServer(Node):
    def __init__(self):
        super().__init__('owl_server')

        # --- Parámetros ---
        self.declare_parameter("image_topic", "/output_camera/image_enhanced")
        self.declare_parameter("model_path", "google/owlv2-base-patch16")
        self.declare_parameter("conf_threshold", 0.20)
        self.declare_parameter("bbox_to_pose_service", "/vision/bbox_to_pose")

        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.model_path = self.get_parameter("model_path").get_parameter_value().string_value
        self.conf_th = self.get_parameter("conf_threshold").get_parameter_value().double_value
        self.bbox_to_pose_service = self.get_parameter("bbox_to_pose_service").get_parameter_value().string_value
        
        self.cb_group = ReentrantCallbackGroup()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.last_image = None

        # --- Publishers ---
        self.debug_pub = self.create_publisher(Image, "/owl/detections", 10)
        self.owl_pub = self.create_publisher(YoloDetections, "/owl/detections_list", 10)
        
        # --- Subscriptions & Clients ---
        self.create_subscription(
            Image, self.image_topic, self.cb_image, 10, callback_group=self.cb_group)

        self.bbox_client = self.create_client(
            BboxToPose, self.bbox_to_pose_service, callback_group=self.cb_group)
        
        while not self.bbox_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'Esperando servicio {self.bbox_to_pose_service}...')

        # --- OWLv2 Model ---
        self.get_logger().info(f"Cargando OWLv2 en {self.device}...")    
        self.processor = Owlv2Processor.from_pretrained(self.model_path)
        self.model = Owlv2ForObjectDetection.from_pretrained(
            self.model_path
        ).to(self.device).half().eval()
        
        # --- Service ---
        self.srv = self.create_service(GetBbox, "/owl/Owlv2Service", self.callback, callback_group=self.cb_group)
        
        self.get_logger().info("OWLv2 Server Listo.")

    def cb_image(self, msg):
        self.last_image = msg
    
    def callback(self, req, resp):
        query = req.class_name.strip().lower()
        
        if query in ["", "all", "everything", "objects"]:
            search_queries = GENERAL_CLASSES
        elif query in CATEGORY:
            search_queries = CATEGORY[query]
        else:
            search_queries = [query]
            
        if self.last_image is None:
            self.get_logger().error("No hay imágenes de la cámara.")
            resp.success = False
            return resp

        cv_img = self.bridge.imgmsg_to_cv2(self.last_image, "bgr8")
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        inputs = self.processor(text=[search_queries], images=img_rgb, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([img_rgb.shape[:2]]).to(self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs, threshold=self.conf_th, target_sizes=target_sizes
        )[0]

        if len(results["boxes"]) == 0:
            self.get_logger().info(f"No se detectó nada para: {query}")
            resp.success = False
            return resp

        keep = torchvision.ops.nms(results["boxes"], results["scores"], iou_threshold=0.5)
        best_idx = keep[0]
        best_box = results["boxes"][best_idx].int().cpu().numpy()
        best_score = results["scores"][best_idx].cpu().item()

        resp.x1, resp.y1, resp.x2, resp.y2 = map(int, best_box)
        resp.confidence = float(best_score)

        # 4. Llamada al servicio de Pose (BboxToPose)
        pose_req = BboxToPose.Request()
        pose_req.x1, pose_req.y1 = resp.x1, resp.y1
        pose_req.x2, pose_req.y2 = resp.x2, resp.y2

        future = self.bbox_client.call_async(pose_req)
        
        start_wait = time.time()
        while not future.done():
            time.sleep(0.01)
            if time.time() - start_wait > 3.0:
                self.get_logger().error("Timeout esperando Pose 3D")
                resp.success = False
                return resp

        pose_res = future.result()
        if pose_res and pose_res.success:
            self.get_logger().info(f"Pose 3D obtenida para {query}")

            t = TransformStamped()
            t.header = pose_res.pose.header
            t.child_frame_id = f"detected_{query[:10]}"
            
            t.transform.translation.x = pose_res.pose.pose.position.x
            t.transform.translation.y = pose_res.pose.pose.position.y
            t.transform.translation.z = pose_res.pose.pose.position.z
            t.transform.rotation = pose_res.pose.pose.orientation
            
            self.tf_broadcaster.sendTransform(t)
            resp.success = True
        else:
            self.get_logger().warn("Fallo al calcular Pose 3D en el nodo C++")
            resp.success = False

        cv2.rectangle(cv_img, (resp.x1, resp.y1), (resp.x2, resp.y2), (0, 255, 0), 2)
        cv2.putText(cv_img, f"{query} {resp.confidence:.2f}", (resp.x1, resp.y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        debug_msg = self.bridge.cv2_to_imgmsg(cv_img, "bgr8")
        self.debug_pub.publish(debug_msg)

        return resp

def main():
    rclpy.init()
    node = OwlServer()
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