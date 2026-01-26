# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Pose, TransformStamped
from vision_interfaces.srv import Owlv2Service
import tf2_ros
import cv2
import numpy as np
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import ros2_numpy


# dictionary of possible aliases for known objects
OBJECT_ALIASES = {
    "coffee mate": [
        "red bottle",
        "plastic bottle",
        "creamer bottle"
    ],
    "sprite": [
        "green bottle",
        "soda can",
        "green can"
    ],
    "pringles": [
        "chips can",
        "cylindrical can",
        "snack can", 
        "potato chips can",
        "red can"
    ],
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

        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.last_cloud = None

        self.debug_pub = self.create_publisher(Image, "/owl/debug_image", 10)

        self.create_subscription(PointCloud2, "/kinect/points", self.cb_points, 10)

        self.get_logger().info("Loading OWLv2...")
        self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16")
        self.model = Owlv2ForObjectDetection.from_pretrained(
            "google/owlv2-base-patch16"
        ).to("cuda").half().eval()
        self.get_logger().info("OWLv2 loaded on GPU")

        self.srv = self.create_service(Owlv2Service, "classifyowl", self.callback)
        self.get_logger().info("OWL service ready.")

    def cb_points(self, msg):
        self.last_cloud = msg

    
    def compute_3d_from_bbox(self, pc, x1, y1, x2, y2):
        try:
            h = self.last_cloud.height
            w = self.last_cloud.width
            pc = pc.reshape((h, w))

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            
            r = 7

            x_min = max(cx - r, 0)
            x_max = min(cx + r, w - 1)
            y_min = max(cy - r, 0)
            y_max = min(cy + r, h - 1)

            xs = pc['x'][y_min:y_max, x_min:x_max].reshape(-1)
            ys = pc['y'][y_min:y_max, x_min:x_max].reshape(-1)
            zs = pc['z'][y_min:y_max, x_min:x_max].reshape(-1)

            mask = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs) & (zs > 0.1) & (zs < 10.0)

            if np.count_nonzero(mask) < 10:
                return None

            mx = float(np.median(xs[mask]))
            my = float(np.median(ys[mask]))
            mz = float(np.median(zs[mask]))

            return mx, my, mz

        except Exception as e:
            self.get_logger().warn(f"3D center sampling failed: {e}")
            return None


    def callback(self, req, resp):
        self.get_logger().info(f"OWL request with {len(req.queries)} queries")

        resp.poses = []
        resp.names = []
        resp.pt_min = []
        resp.pt_max = []
        resp.confidence = []
        resp.debug_images = []

        if req.image.width == 0:
            return resp

        cv_img = self.bridge.imgmsg_to_cv2(req.image, "bgr8")
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        texts = [req.queries]

        inputs = self.processor(text=texts, images=img_rgb, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([img_rgb.shape[:2]]).to("cuda")
        results = self.processor.post_process_object_detection(
            outputs, threshold=0.2, target_sizes=target_sizes
        )[0]

        debug_img = img_rgb.copy()

        if len(results["boxes"]) == 0:
            self.get_logger().info("No detections")
            debug_bgr = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)
            msg = self.bridge.cv2_to_imgmsg(debug_bgr, encoding="bgr8")
            self.debug_pub.publish(msg)
            resp.debug_images.append(msg)
            return resp
        
        best_per_class = {}

        for i, (box, score, label) in enumerate(zip(results["boxes"], results["scores"], results["labels"])):
            conf = float(score.cpu())
            cls = int(label.cpu())

            if conf < 0.3:
                continue
            if cls not in best_per_class or conf > best_per_class[cls]["conf"]:
                best_per_class[cls] = {
                    "box": box,
                    "conf": conf
                }
        if len(best_per_class) == 0:
            self.get_logger().info("No detections after filtering")
            return resp
        
        for cls, data in best_per_class.items():
            box = data["box"]
            conf = data["conf"]

            x1, y1, x2, y2 = box.int().cpu().numpy()
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            raw_name = req.queries[int(cls)]
            class_name = normalize_label(raw_name)

            cv2.rectangle(debug_img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(debug_img, f"{class_name} {conf:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.circle(debug_img, (cx, cy), 5, (0,0,255), -1)

            p = Pose()
            valid_3d = False

            if self.last_cloud is not None:
                pc = ros2_numpy.numpify(self.last_cloud)
                result = self.compute_3d_from_bbox(pc, x1, y1, x2, y2)

                if result is not None:
                    mx, my, mz = result
                    p.position.x = mx
                    p.position.y = my
                    p.position.z = mz
                    p.orientation.w = 1.0
                    valid_3d = True

                    # TFs
                    t = TransformStamped()
                    t.header.frame_id = self.last_cloud.header.frame_id
                    t.child_frame_id = f"owl_{class_name}_{i}"
                    t.header.stamp = self.get_clock().now().to_msg()
                    t.transform.translation.x = mx
                    t.transform.translation.y = my
                    t.transform.translation.z = mz
                    t.transform.rotation.w = 1.0

                    self.tf_broadcaster.sendTransform(t)

            if not valid_3d:
                self.get_logger().warn(f"No valid depth for {class_name}")

            resp.poses.append(p)
            resp.names.append(class_name)
            resp.pt_min.extend([float(x1), float(y1)])
            resp.pt_max.extend([float(x2), float(y2)])
            resp.confidence.append(conf)
            resp.cx.extend([float((x1 + x2) / 2)])
            resp.cy.extend([float((y1 + y2) / 2)])

        debug_bgr = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)
        msg = self.bridge.cv2_to_imgmsg(debug_bgr, encoding="bgr8")
        self.debug_pub.publish(msg)
        resp.debug_images.append(msg)

        return resp


def main():
    rclpy.init()
    node = OwlServer()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()



