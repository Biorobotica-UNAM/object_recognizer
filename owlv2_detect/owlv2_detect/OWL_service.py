#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import threading
import time

import cv2
import numpy as np
import rclpy
import torch
import torchvision
import tf2_geometry_msgs  # noqa: F401
from cv_bridge import CvBridge
from geometry_msgs.msg import Quaternion, TransformStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from vision_interfaces.msg import YoloDetection, YoloDetections
from vision_interfaces.srv import GetDetections, BboxToPose

CATEGORY = {
    "drinks": ["coke can", "red bull can", "coke zero can", "pepsi bottle", "milk carton", "soju bottle"],
    "snacks": ["pringles", "seaweed snack"],
    "fabric": ["blue shirt", "grey shirt", "white shirt", "black shirt", "hand towel"],
    "food": ["blue container cornflakes", "instant noodles", "black container cornflakes"],
    "fruits": ["red apple", "mangostane", "red bellpepper", "peach", "yellow lemon", "yellow bellpepper"],
    "trash": ["red apple", "mangostane", "red bellpepper", "peach", "yellow lemon", "yellow bellpepper"],
    "cutlery": ["spoon", "fork", "knife", "chopsticks", "metal spoon", "plastic fork"],
    "cleaning_supply": ["sponge", "dishwasher tablet", "toothpaste"],
    "containers": ["backpack", "handbag", "suitcase", "box", "basket", "dishwasher", "washing machine"],
    # container_drop_tf_node .
    "container_drop": ["bucket", "red bucket", "trash can", "waste basket", "plastic bin", "container", "open container", "storage bin", "box", "basket"],
    "electrodomestics": ["fridge", "oven", "microwave", "toaster", "kettle", "coffee maker"],
    "toys": ["rubiks cube"],
    "tableware": ["red bowl", "red plate", "mug", "cup","bowl", "plastic bowl", "plastic mug", "glass cup"],

    "groceries": ["coke can", "red bull can", "coke zero can", "pepsi bottle",
                  "milk carton", "soju bottle", "pringles", "seaweed snack","blue shirt", "grey shirt",
                  "white shirt", "black shirt", "hand towel",
                  "blue container cornflakes", "instant noodles", "black container cornflakes",
                  "sponge", "dishwasher tablet", "toothpaste",
                  "rubiks cube" ]
}

GENERAL_CLASSES = [
    "bottled water", "soda can", "juice box", "milk carton", "coffee mug",
    "glass cup", "paper plate", "plastic bowl", "metal spoon", "fork",
    "knife", "apple", "banana", "orange", "cereal box", "snack bag",
    "chips", "bread", "sponge", "soap bottle", "trash can", "backpack",
    "laptop", "cell phone", "book", "remote control", "keys", "person",
    "bottle", "woman", "robot", "chair", "sofa", "dining table", "shelf", "green mug", "purple mug",
    "pink bowl", "purple bowl", "pink cup"
]

class OwlServer(Node):
    def __init__(self):
        super().__init__("owl_server")
        defaults = {
            "image_topic": "hardware/camera/image_raw",
            "model_path": "google/owlv2-base-patch16",
            "bbox_to_pose_service": "/vision/bbox_to_pose",
            "conf_threshold": 0.32, "nms_iou_class": 0.3, "nms_iou_any": 0.7,
            "pre_white_balance": False, "pre_denoise": False, "pre_clahe": False,
            "clahe_clip": 2.0, "clahe_grid": 8, "publish_preprocessed": False,
            "max_image_age_sec": 2.0, "pose_timeout_sec": 1.0,
            "tf_ttl_sec": 2.0, "target_frame": "base_link",
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.bridge = CvBridge()
        self._img_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._tf_lock = threading.Lock()
        self.last_cv_image = self.last_header = None
        self.last_tfs = {}
        self.cb_group = ReentrantCallbackGroup()
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(0.1, self.expire_tfs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Cargando OWLv2 en {self.device}...")
        model_path = self.get_parameter("model_path").value
        self.processor = Owlv2Processor.from_pretrained(model_path)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_path).to(self.device)
        if self.device == "cuda":
            self.model.half()
        self.model.eval()
        self.dtype = next(self.model.parameters()).dtype
        self.pose_client = self.create_client(
            BboxToPose, self.get_parameter("bbox_to_pose_service").value,
            callback_group=self.cb_group)
        self.create_subscription(
            Image, self.get_parameter("image_topic").value, self.cb_image,
            qos_profile_sensor_data, callback_group=self.cb_group)
        self.det_pub = self.create_publisher(YoloDetections, "/vision/detections", 10)
        self.debug_pub = self.create_publisher(Image, "/vision/owl/debug_image", 10)
        self.pre_pub = self.create_publisher(Image, "/vision/owl/preprocessed_image", 10)
        self.srv = self.create_service(
            GetDetections, "owl/owlv2_service", self.handle_analyze,
            callback_group=self.cb_group)

    def cb_image(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self._img_lock:
                self.last_cv_image = image
                self.last_header = copy.deepcopy(msg.header)
        except Exception as exc:
            self.get_logger().error(f"Imagen inválida: {exc}")

    def expire_tfs(self):
        # Nunca rejuvenecer una observación cambiando su timestamp.
        # La caché local caduca; TF2 conserva su propio historial.
        now = time.monotonic()
        with self._tf_lock:
            self.last_tfs = {key: value for key, value in self.last_tfs.items()
                             if value[1] > now}

    def preprocess(self, image):
        out = image
        if self.get_parameter("pre_white_balance").value:
            f = out.astype(np.float32)
            means = f.reshape(-1, 3).mean(axis=0) + 1e-6
            out = np.clip(f * (means.mean() / means), 0, 255).astype(np.uint8)
        if self.get_parameter("pre_denoise").value:
            out = cv2.bilateralFilter(out, 5, 40, 40)
        if self.get_parameter("pre_clahe").value:
            grid = max(1, int(self.get_parameter("clahe_grid").value))
            clahe = cv2.createCLAHE(
                clipLimit=max(0.01, float(self.get_parameter("clahe_clip").value)),
                tileGridSize=(grid, grid))
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return out

    def _infer(self, image, queries):
        h, w = image.shape[:2]
        with self._infer_lock:
            inputs = self.processor(text=[queries], images=cv2.cvtColor(
                image, cv2.COLOR_BGR2RGB), return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
            with torch.inference_mode():
                outputs = self.model(**inputs)
                outputs.logits = outputs.logits.float()
                outputs.pred_boxes = outputs.pred_boxes.float()
                # OWLv2 normalmente aplica padding inferior/derecho a cuadrado.
                padded = getattr(self.processor.image_processor, "do_pad", True)
                size = [max(h, w)] * 2 if padded else [h, w]
                result = self.processor.post_process_grounded_object_detection(
                    outputs, threshold=float(self.get_parameter("conf_threshold").value),
                    target_sizes=torch.tensor([size], device=self.device))[0]
                boxes, scores, labels = result["boxes"], result["scores"], result["labels"]
                boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
                boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)
                valid = (torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores)
                         & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]))
                boxes, scores, labels = boxes[valid], scores[valid], labels[valid]
                keep = torchvision.ops.batched_nms(
                    boxes, scores, labels, float(self.get_parameter("nms_iou_class").value))
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                keep = torchvision.ops.nms(
                    boxes, scores, float(self.get_parameter("nms_iou_any").value))
                return (boxes[keep].cpu().numpy(), scores[keep].cpu().numpy(),
                        labels[keep].cpu().numpy())

    async def get_pose(self, request):
        future = self.pose_client.call_async(request)
        timeout = max(0.01, float(self.get_parameter("pose_timeout_sec").value))
        timer = self.create_timer(timeout, future.cancel, callback_group=self.cb_group)
        try:
            return await future
        finally:
            timer.cancel()
            self.destroy_timer(timer)

    async def handle_analyze(self, req, resp):
        try:
            return await self._analyze(req, resp)
        except Exception as exc:
            self.get_logger().error(f"Error en detección OWL: {exc}")
            resp.success = False
            resp.message = f"Error en detección: {exc}"
            return resp

    async def _analyze(self, req, resp):
        queries = []
        for text in req.queries or ["object"]:
            text = text.lower().strip()
            queries.extend(CATEGORY.get(text, GENERAL_CLASSES if text in (
                "all", "everything") else [text]))
        queries = list(dict.fromkeys(text for text in queries if text))
        if not queries:
            resp.success, resp.message = False, "Las consultas están vacías"
            return resp
        with self._img_lock:
            if self.last_cv_image is None:
                resp.success, resp.message = False, "No hay imagen disponible"
                return resp
            image, header = self.last_cv_image.copy(), copy.deepcopy(self.last_header)
        resp.header = header
        stamp_ns = header.stamp.sec * 10**9 + header.stamp.nanosec
        age = (self.get_clock().now().nanoseconds - stamp_ns) / 1e9
        max_age = float(self.get_parameter("max_image_age_sec").value)
        if stamp_ns == 0 or (max_age > 0 and (age < 0 or age > max_age)):
            resp.success, resp.message = False, "Imagen sin timestamp válido o demasiado antigua"
            return resp
        model_image = self.preprocess(image)
        if self.get_parameter("publish_preprocessed").value:
            msg = self.bridge.cv2_to_imgmsg(model_image, "bgr8")
            msg.header = header
            self.pre_pub.publish(msg)
        boxes, scores, labels = self._infer(model_image, queries)
        array = YoloDetections()
        array.header = header
        counters, pose_count = {}, 0
        h, w = image.shape[:2]
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(float, box)
            name = queries[int(label)]
            frame_name = name.replace(" ", "_")
            instance = counters.get(frame_name, 0)
            counters[frame_name] = instance + 1
            detection = YoloDetection()
            detection.class_name, detection.confidence = name, float(score)
            detection.x_min, detection.y_min = x1, y1
            detection.x_max, detection.y_max = x2, y2
            array.detections.append(detection)
            if self.pose_client.service_is_ready():
                request = BboxToPose.Request()
                request.header = header
                request.x1, request.y1 = int(max(0, x1 - 5)), int(max(0, y1 - 5))
                request.x2, request.y2 = int(min(w, np.ceil(x2 + 5))), int(min(h, np.ceil(y2 + 5)))
                try:
                    result = await self.get_pose(request)
                    if result is not None and result.success:
                        pose = self.tf_buffer.transform(
                            result.pose, self.get_parameter("target_frame").value)
                        vertical = result.pose.pose.orientation.y < -0.5
                        rotation = Quaternion()
                        rotation.w = 0.5 if vertical else 0.9659258
                        rotation.y = -0.8660254 if vertical else -0.258819
                        transform = TransformStamped()
                        transform.header = copy.deepcopy(pose.header)
                        transform.child_frame_id = f"OWL_{frame_name}_{instance}"
                        transform.transform.translation.x = pose.pose.position.x
                        transform.transform.translation.y = pose.pose.position.y
                        transform.transform.translation.z = pose.pose.position.z
                        transform.transform.rotation = rotation
                        pose_stamp = transform.header.stamp
                        pose_age = (self.get_clock().now().nanoseconds - (
                            pose_stamp.sec * 10**9 + pose_stamp.nanosec)) / 1e9
                        ttl = float(self.get_parameter("tf_ttl_sec").value)
                        if ttl > 0 and 0 <= pose_age < ttl:
                            self.tf_broadcaster.sendTransform(transform)
                            with self._tf_lock:
                                self.last_tfs[transform.child_frame_id] = (
                                    transform, time.monotonic() + ttl - pose_age)
                            pose_count += 1
                    else:
                        self.get_logger().warning(f"Sin pose 3D para {name} (fallo o timeout)")
                except Exception as exc:
                    self.get_logger().warning(f"Pose 3D no disponible para {name}: {exc}")
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(image, p1, p2, (0, 255, 0), 2)
            cv2.putText(image, f"OWL_{name}_{instance}", (p1[0], max(12, p1[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Publicar también resultados vacíos para limpiar consumidores 2D.
        self.det_pub.publish(array)
        debug = self.bridge.cv2_to_imgmsg(image, "bgr8")
        debug.header = header
        self.debug_pub.publish(debug)
        resp.detections = array.detections
        resp.success = bool(array.detections)
        resp.message = (f"{len(array.detections)} detecciones 2D; {pose_count} poses 3D"
                        if resp.success else "No se detectó nada")
        return resp


def main():
    rclpy.init()
    node = OwlServer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
