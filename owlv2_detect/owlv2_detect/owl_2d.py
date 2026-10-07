#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading

import cv2
import numpy as np
import rclpy
import torch
import torchvision
from cv_bridge import CvBridge
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor
from vision_interfaces.msg import YoloDetection, YoloDetections
from vision_interfaces.srv import GetDetections

# VERSION ESTABLE - SOLO 2D (sin TF / base_link)
# + preprocesado clasico (white balance, denoise, CLAHE) solo para la entrada del modelo

CATEGORY = {
    "drinks": ["coke can", "red bull can", "coke zero can", "pepsi bottle", "milk carton", "soju bottle", "soju", "cardboard milk", "milk whit coffe cardboard"],
    "snacks": ["pringles", "seaweed snack", "seaweed", "green wrap", "grean seaweed wrap"],
    "fabric": ["blue shirt", "grey shirt", "white shirt", "black shirt", "hand towel"],
    "food": ["blue container cornflakes", "instant noodles", "red wrapper", "brown container cornflakes", "cornflakes box", "cornflakes", "brown cornflakes box", "brown cornflakes"],
    "fruits": ["red apple", "mangostane", "red bellpepper", "peach", "yellow lemon", "yellow bellpepper"],
    "cutlery": ["spoon", "fork", "knife", "chopsticks", "metal spoon", "plastic fork"],
    "cleaning_supply": ["sponge", "dishwasher tablet", "toothpaste"],
    "containers": ["backpack", "handbag", "suitcase", "box", "basket", "dishwasher", "washing machine"],
    # container_drop_tf_node .
    "container_drop": ["bucket", "red bucket", "trash can", "waste basket", "plastic bin", "container", "open container", "storage bin", "box", "basket"],
    "electrodomestics": ["fridge", "oven", "microwave", "toaster", "kettle", "coffee maker"],
    "toys": ["rubiks cube"],
    "tableware": ["red bowl", "red plate", "mug", "cup", "bowl", "plastic bowl", "plastic mug", "glass cup"],
}

GENERAL_CLASSES = [
    "bottled water", "soda can", "juice box", "milk carton", "coffee mug",
    "glass cup", "paper plate", "plastic bowl", "metal spoon", "fork",
    "knife", "apple", "banana", "orange", "cereal box", "snack bag",
    "chips", "bread", "sponge", "soap bottle", "trash can", "backpack",
    "laptop", "cell phone", "book", "remote control", "keys", "person",
    "bottle", "woman", "robot", "chair", "sofa", "dining table", "shelf", "green mug", "purple mug",
    "pink bowl", "purple bowl", "pink cup",
]


class OwlServer(Node):
    def __init__(self):
        super().__init__('owl_server')

        # --- Parámetros ---
        self.declare_parameter("image_topic", "/camera/rgb/image_raw")
        self.declare_parameter("model_path", "google/owlv2-base-patch16")
        self.declare_parameter("conf_threshold", 0.28)

        # NMS: primero por clase, luego una pasada agnóstica con IoU alto
        # (esta última solo quita duplicados de sinónimos: "cornflakes" vs "brown cornflakes")
        self.declare_parameter("nms_iou_class", 0.3)
        self.declare_parameter("nms_iou_any", 0.7)

        # Preprocesado clásico (se pueden cambiar en caliente con `ros2 param set`)
        self.declare_parameter("pre_white_balance", True)
        self.declare_parameter("pre_denoise", True)
        self.declare_parameter("pre_clahe", True)
        self.declare_parameter("clahe_clip", 2.0)
        self.declare_parameter("clahe_grid", 8)
        self.declare_parameter("publish_preprocessed", True)

        self.topic_name = self.get_parameter("image_topic").get_parameter_value().string_value
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bridge = CvBridge()

        self.last_cv_image = None
        self.last_header = None
        self._img_lock = threading.Lock()    # protege last_cv_image / last_header
        self._infer_lock = threading.Lock()  # una sola inferencia a la vez en la GPU
        self.cb_group = ReentrantCallbackGroup()

        # --- Modelo OWLv2 ---
        model_path = self.get_parameter("model_path").value
        self.get_logger().info(f"Cargando OWLv2 en {self.device}...")
        self.processor = Owlv2Processor.from_pretrained(model_path)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_path).to(self.device)

        if self.device == "cuda":
            self.model = self.model.half()
        self.model.eval()
        self.dtype = next(self.model.parameters()).dtype

        # --- Subs, Pubs y Servicio ---
        self.create_subscription(Image, self.topic_name, self.cb_image, 10,
                                 callback_group=self.cb_group)

        self.det_pub = self.create_publisher(YoloDetections, "/vision/detections", 10)
        self.debug_pub = self.create_publisher(Image, "/vision/owl/debug_image_2d", 10)
        self.pre_pub = self.create_publisher(Image, "/vision/owl/preprocessed_image", 10)

        self.srv = self.create_service(GetDetections, "owl/owlv2_service",
                                       self.handle_analyze, callback_group=self.cb_group)

        self.get_logger().info(f"OWLv2 operativo. Escuchando {self.topic_name}")

    # ------------------------------------------------------------------
    # Imagen
    # ------------------------------------------------------------------
    def cb_image(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        with self._img_lock:
            self.last_cv_image = cv_img
            self.last_header = msg.header

    # ------------------------------------------------------------------
    # Preprocesado clásico (solo para la entrada del modelo)
    # ------------------------------------------------------------------
    @staticmethod
    def _gray_world(bgr):
        """White balance gray-world: iguala la media de los 3 canales."""
        f = bgr.astype(np.float32)
        means = f.reshape(-1, 3).mean(axis=0) + 1e-6
        gain = means.mean() / means
        return np.clip(f * gain, 0, 255).astype(np.uint8)

    def preprocess(self, bgr):
        """
        Orden: white balance -> denoise -> CLAHE.
        El denoise va antes de CLAHE porque CLAHE amplifica el ruido.
        """
        out = bgr

        if self.get_parameter("pre_white_balance").value:
            out = self._gray_world(out)

        if self.get_parameter("pre_denoise").value:
            out = cv2.bilateralFilter(out, 5, 40, 40)

        if self.get_parameter("pre_clahe").value:
            grid = int(self.get_parameter("clahe_grid").value)
            clahe = cv2.createCLAHE(
                clipLimit=float(self.get_parameter("clahe_clip").value),
                tileGridSize=(grid, grid),
            )
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            out = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)

        return out

    # ------------------------------------------------------------------
    # Servicio (def normal: no hay ningún await, async no aportaba nada)
    # ------------------------------------------------------------------
    def handle_analyze(self, req, resp):
        # --- Queries ---
        raw_queries = req.queries if req.queries else ["object"]
        final_queries = []
        for q in raw_queries:
            q_lower = q.lower().strip()
            if q_lower in CATEGORY:
                final_queries.extend(CATEGORY[q_lower])
            elif q_lower in ("all", "everything"):
                final_queries.extend(GENERAL_CLASSES)
            else:
                final_queries.append(q_lower)
        # quita duplicados (varias categorías comparten "box", "basket"...) conservando el orden
        final_queries = list(dict.fromkeys(q.strip() for q in final_queries if q.strip()))

        with self._img_lock:
            if self.last_cv_image is None:
                resp.success = False
                resp.message = "Error: No hay imagen disponible"
                return resp
            cv_img = self.last_cv_image.copy()
            header = self.last_header

        resp.header = header
        h, w = cv_img.shape[:2]

        # --- Preprocesado: solo la entrada del modelo; el debug se dibuja sobre la original ---
        model_in = self.preprocess(cv_img)
        if self.get_parameter("publish_preprocessed").value:
            pre_msg = self.bridge.cv2_to_imgmsg(model_in, "bgr8")
            pre_msg.header = header
            self.pre_pub.publish(pre_msg)

        img_rgb = cv2.cvtColor(model_in, cv2.COLOR_BGR2RGB)

        # --- Inferencia ---
        with self._infer_lock:
            inputs = self.processor(text=[final_queries], images=img_rgb, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            # FIX: con model.half() pixel_values debe ir en el mismo dtype (input_ids se queda en long)
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # post-proceso en fp32 (fp16 pierde precisión en coordenadas)
            outputs.logits = outputs.logits.float()
            outputs.pred_boxes = outputs.pred_boxes.float()

            # FIX: OWLv2 pad-ea la imagen a cuadrado (abajo/derecha), las cajas son relativas
            # al cuadrado -> se escala con max(h, w) en ambos lados, no con (h, w)
            side = max(h, w)
            target_sizes = torch.tensor([[side, side]], device=self.device)
            results = self.processor.post_process_grounded_object_detection(
                outputs, threshold=self.get_parameter("conf_threshold").value,
                target_sizes=target_sizes
            )[0]

        boxes, scores, labels = results["boxes"], results["scores"], results["labels"]

        if len(boxes) == 0:
            resp.success = False
            resp.message = "No se detectó nada"
            return resp

        # --- NMS en 2 etapas ---
        iou_cls = float(self.get_parameter("nms_iou_class").value)
        iou_any = float(self.get_parameter("nms_iou_any").value)

        keep = torchvision.ops.batched_nms(boxes, scores, labels, iou_cls)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        keep = torchvision.ops.nms(boxes, scores, iou_any)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # recorta a la imagen real (las cajas pueden caer en la zona de padding)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)

        boxes_np = boxes.cpu().numpy()
        scores_np = scores.cpu().numpy()
        labels_np = labels.cpu().numpy()

        # --- Mensajes ---
        det_array = YoloDetections()
        det_array.header = header
        class_counter = {}

        for box, score, label in zip(boxes_np, scores_np, labels_np):
            x1, y1, x2, y2 = (float(v) for v in box)
            if x2 <= x1 or y2 <= y1:  # caja degenerada tras el clip
                continue

            raw_class_name = final_queries[int(label)]  # FIX: índice int
            class_name = raw_class_name.replace(" ", "_")
            instance_id = class_counter.get(class_name, -1) + 1
            class_counter[class_name] = instance_id

            d = YoloDetection()
            d.class_name = raw_class_name
            d.confidence = float(score)
            d.x_min, d.y_min, d.x_max, d.y_max = x1, y1, x2, y2
            det_array.detections.append(d)

            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(cv_img, p1, p2, (0, 255, 0), 2)
            cv2.putText(cv_img, f"OWL_{raw_class_name}_{instance_id}", (p1[0], max(p1[1] - 10, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if not det_array.detections:
            resp.success = False
            resp.message = "No se detectó nada"
            return resp

        debug_msg = self.bridge.cv2_to_imgmsg(cv_img, "bgr8")
        debug_msg.header = header
        self.debug_pub.publish(debug_msg)

        resp.detections = det_array.detections
        resp.success = True
        self.det_pub.publish(det_array)
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