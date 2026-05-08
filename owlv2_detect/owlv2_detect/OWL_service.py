# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from vision_interfaces.srv import GetDetections, BboxToPose
from vision_interfaces.msg import YoloDetection, YoloDetections
import cv2
import torch
import torchvision
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from tf2_ros import TransformBroadcaster, TransformListener, Buffer
import tf2_geometry_msgs
from geometry_msgs.msg import TransformStamped

CATEGORY = {
    "trash": ["bottled water", "soda can", "juice box", "milk carton", "sprite", "coke", "sprite can", "fanta can", "red can", "milk bottle"],
    "snacks": ["chips", "biscuit pack", "snack bag", "pringles", "popcorn"],
    "fruits": ["apple", "banana", "orange", "peach", "pear", "strawberry", "lemon"],
    "cuttlery": ["plastic bowl", "fork", "knife"],
    "containers": ["backpack", "handbag", "suitcase", "box", "basket", "dishwasher", "washing machine"],
    "electrodomestics": ["fridge", "oven", "microwave", "toaster", "kettle", "coffee maker"],
    "toys": ["lego", "doll", "action figure", "board game", "puzzle", "teddy bear"],
    "tableware": ["red plate","mug", "cup"],
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
        self.declare_parameter("image_topic", "/kinect/image_raw")
        self.declare_parameter("model_path", "google/owlv2-base-patch16")
        self.declare_parameter("conf_threshold", 0.28)

        self.topic_name = self.get_parameter("image_topic").get_parameter_value().string_value
        self.conf_th = self.get_parameter("conf_threshold").value
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bridge = CvBridge()
        
        self.last_cv_image = None
        self.last_header = None
        self.cb_group = ReentrantCallbackGroup()
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)


        # --- Modelo OWLv2 ---
        self.get_logger().info(f"Cargando OWLv2 en {self.device}...")
        self.processor = Owlv2Processor.from_pretrained(self.get_parameter("model_path").value)
        self.model = Owlv2ForObjectDetection.from_pretrained(
            self.get_parameter("model_path").value
        ).to(self.device).half().eval()

        # --- Clientes y Pubs ---
        self.pose_client = self.create_client(BboxToPose, "bbox_to_pose", callback_group=self.cb_group)
        self.create_subscription(Image, self.topic_name, self.cb_image, 10, callback_group=self.cb_group)
        
        self.det_pub = self.create_publisher(YoloDetections, "/vision/detections", 10)
        self.debug_pub = self.create_publisher(Image, "/vision/owl/debug_image", 10)
        
        self.srv = self.create_service(GetDetections, "owl/owlv2_service", 
                                       self.handle_analyze, callback_group=self.cb_group)
        
        self.get_logger().info(f"OWLv2 operativo. Escuchando {self.topic_name}")

    def cb_image(self, msg):
        self.last_cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.last_header = msg.header

    async def handle_analyze(self, req, resp):

        raw_queries = req.queries if req.queries else ["object"]
        final_queries = []
        for q in raw_queries:
            q_lower = q.lower().strip()
            if q_lower in CATEGORY: final_queries.extend(CATEGORY[q_lower])
            elif q_lower in ["all", "everything"]: final_queries.extend(GENERAL_CLASSES)
            else: final_queries.append(q_lower)

        if self.last_cv_image is None:
            resp.success = False
            resp.message = "Error: No hay imagen disponible"
            return resp

        cv_img = self.last_cv_image.copy()
        header = self.last_header

        # iferencia
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        inputs = self.processor(text=[final_queries], images=img_rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([img_rgb.shape[:2]]).to(self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs, threshold=self.conf_th, target_sizes=target_sizes
        )[0]

       
        if len(results["boxes"]) > 0:
            keep = torchvision.ops.nms(results["boxes"], results["scores"], iou_threshold=0.5)
            det_array = YoloDetections()
            det_array.header = header

            for idx in keep:
                box = results["boxes"][idx].int().cpu().numpy()
                d = YoloDetection()
                d.class_name = final_queries[results["labels"][idx]]
                d.confidence = float(results["scores"][idx].cpu().item())
                d.x_min, d.y_min, d.x_max, d.y_max = map(float, box)


                if self.pose_client.service_is_ready():
                    pose_req = BboxToPose.Request()
                    m = 5 # Margen pequeño
                    pose_req.x1, pose_req.y1 = int(max(0, box[0]-m)), int(max(0, box[1]-m))
                    pose_req.x2, pose_req.y2 = int(box[2]+m), int(box[3]+m)
                    
                    result_3d = await self.pose_client.call_async(pose_req)
                    
                    if result_3d.success:
                        try:
                            transform = self.tf_buffer.lookup_transform(
                                "base_link",
                                 result_3d.pose.header.frame_id,
                                 rclpy.time.Time(),
                                 timeout=rclpy.duration.Duration(seconds=1.0)
                            )
                            pose_in_base = tf2_geometry_msgs.do_transform_pose(result_3d.pose, transform)
                            
                            t = TransformStamped()
                            t.header.stamp = self.get_clock().now().to_msg()
                            t.header.frame_id = "base_link"
                            clean_name = d.class_name.replace(" ", "_")
                            t.child_frame_id = f"OWL_{clean_name}"
                            
                            t.transform.translation.x = pose_in_base.pose.position.x
                            t.transform.translation.y = pose_in_base.pose.position.y
                            t.transform.translation.z = pose_in_base.pose.position.z
                            t.transform.rotation = pose_in_base.pose.orientation
                        
                            self.tf_broadcaster.sendTransform(t)
                            self.get_logger().info(f"TF publicada: {t.child_frame_id}")

                        except Exception as e:
                            self.get_logger().error(f"Error al transformar pose: {e}")

                det_array.detections.append(d)
                cv2.rectangle(cv_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(cv_img, f"OWL_{d.class_name}", (box[0], box[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            debug_msg = self.bridge.cv2_to_imgmsg(cv_img, "bgr8")
            debug_msg.header = header
            self.debug_pub.publish(debug_msg)
            
            resp.detections = det_array.detections
            resp.success = True
            self.det_pub.publish(det_array)
        else:
            resp.success = False
            resp.message = "No se detectó nada"

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