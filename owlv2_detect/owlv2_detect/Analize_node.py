#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from vision_interfaces.msg import YoloDetections, YoloDetection
from vision_interfaces.srv import GetAnalytics, GetDetections


class VisionAnalyticService(Node):
    def __init__(self):
        super().__init__('vision_analytic_service')
        self.cb_group = ReentrantCallbackGroup()

        self.declare_parameter("iou_threshold", 0.3)
        self.iou_threshold = self.get_parameter("iou_threshold").value

        self.last_detections = []
        self.cumulative_count = 0
        self.object_history = {}   # {id: bounding_box}
        self.next_id = 0

        self.owl_client = self.create_client(
            GetDetections,
            '/vision/owl/owlv2_service',
            callback_group=self.cb_group
        )

        self.sub = self.create_subscription(
            YoloDetections, "/vision/detections", self.cache_callback, 10
        )

        service_map = [
            '/vision/count_objects',
            '/vision/get_by_size',
            '/vision/get_by_color',
            '/vision/compare_count'
        ]
        for srv_name in service_map:
            self.create_service(GetAnalytics, srv_name, self.master_handler, callback_group=self.cb_group)

        self.get_logger().info("Node 'vision_analytic_service' initialized and services ready.")

    async def master_handler(self, request, response):
        action = request.action.strip().lower()

        if action == "count":
            return await self.handle_count(request, response)
        elif action in ("size", "get_by_size"):
            return await self.handle_size(request, response)
        elif action in ("color", "get_by_color"):
            return await self.handle_color(request, response)
        elif action in ("compare", "compare_count"):
            return await self.handle_compare(request, response)
        else:
            response.success = False
            response.message = f"Action '{action}' invalid. Use: count | get_by_size | get_by_color | compare_count."
            self.get_logger().error(response.message)
            return response

    def cache_callback(self, msg):
        self.last_detections = msg.detections

    def _check_service(self, client):
        if not client.service_is_ready():
            client.wait_for_service(timeout_sec=2.0)
        return client.service_is_ready()

    async def _call_owl(self, queries: list):
        if not self._check_service(self.owl_client):
            self.get_logger().error("OWLv2 service offline.")
            return False

        owl_req = GetDetections.Request()
        owl_req.queries = queries
        self.get_logger().info(f"Calling OWLv2 with queries: {queries}")

        owl_resp = await self.owl_client.call_async(owl_req)

        if owl_resp.success and len(owl_resp.detections) > 0:
            self.last_detections = owl_resp.detections
            return True

        self.last_detections = []
        return False

    def calculate_iou(self, boxA, boxB):
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        union = areaA + areaB - interArea
        return interArea / float(union) if union > 0 else 0

    def find_match(self, new_box):
        best_id, max_iou = None, self.iou_threshold
        for obj_id, old_box in self.object_history.items():
            iou = self.calculate_iou(new_box, old_box)
            if iou > max_iou:
                max_iou, best_id = iou, obj_id
        return best_id

    def _process_detections(self, target_class: str = "", reset_count: bool = False):
        if reset_count:
            self.cumulative_count = 0
            self.object_history = {}
            self.next_id = 0
            self.get_logger().info("Counter reset.")

        all_processed = []
        target_only = []

        for det in self.last_detections:
            box = [det.x_min, det.y_min, det.x_max, det.y_max]
            area = (det.x_max - det.x_min) * (det.y_max - det.y_min)

            match_id = self.find_match(box)
            if match_id is None:
                match_id = self.next_id
                self.next_id += 1
                self.cumulative_count += 1

            self.object_history[match_id] = box

            new_det = YoloDetection()
            new_det.x_min, new_det.y_min = det.x_min, det.y_min
            new_det.x_max, new_det.y_max = det.x_max, det.y_max
            new_det.confidence = det.confidence
            new_det.class_name = f"ID_{match_id}_{det.class_name}"

            obj_data = {'msg': new_det, 'area': area, 'raw_label': det.class_name, 'confidence': det.confidence}
            all_processed.append(obj_data)

            if not target_class or target_class.lower() in det.class_name.lower():
                target_only.append(obj_data)

        target_only.sort(key=lambda x: x['area'], reverse=True)
        return all_processed, target_only

    async def handle_count(self, request, response):
        target = request.target_class.strip().lower()

        if not await self._call_owl([target] if target else ["all"]):
            response.success = False
            response.message = f"OWLv2 found no detections for '{target}'."
            return response

        _, target_only = self._process_detections(
            target_class=target,
            reset_count=request.reset_count
        )
        response.success     = True
        response.class_count = len(target_only)
        response.message     = f"Found {len(target_only)} '{target}'."
        self.get_logger().info(response.message)
        return response

    async def handle_size(self, request, response):
        mode   = request.filter_mode.strip().lower()
        target = request.target_class.strip().lower()

        if mode not in ("big", "small"):
            response.success = False
            response.message = f"filter_mode '{mode}' invalid. Use: big | small."
            return response

        queries = [target] if target else ["all"]
        if not await self._call_owl(queries):
            response.success = False
            response.message = f"OWLv2 found no detections for '{queries}'."
            return response

        _, target_only = self._process_detections(target_class=target)
        if not target_only:
            response.success = False
            response.message = f"No objects of class '{target}' found."
            return response

        best = target_only[0] if mode == "big" else target_only[-1]

        response.success         = True
        response.best_match      = best['msg']
        response.best_match_area = float(best['area'])
        response.message         = f"'{mode}' of '{target}': {best['msg'].class_name}."
        self.get_logger().info(response.message)
        return response

    async def handle_color(self, request, response):
        color  = request.color.strip().lower()
        target = request.target_class.strip().lower()

        if not color or not target:
            response.success = False
            response.message = "Fields 'color' and 'target_class' are required for color action."
            return response

        color_query = f"{color} {target}"

        if not await self._call_owl([color_query]):
            response.success = False
            response.message = f"OWLv2 found no detections for '{color_query}'."
            return response

        _, color_only = self._process_detections(target_class=color_query.replace(" ", "_"))
        if not color_only:
            _, color_only = self._process_detections(target_class=color_query)

        if not color_only:
            response.success = False
            response.message = f"No objects '{color_query}' found after processing."
            return response

        best = max(color_only, key=lambda x: x['confidence'])
        response.success = True
        response.best_match = best['msg']
        response.message = f"Found '{color_query}': {best['msg'].class_name} (conf={best['confidence']:.2f})."
        self.get_logger().info(response.message)
        return response

    async def handle_compare(self, request, response):
        classes = [c.strip().lower() for c in request.compare_classes if c.strip()]

        if len(classes) < 2:
            response.success = False
            response.message = "Need at least 2 classes to compare."
            return response

        if not await self._call_owl(classes):
            response.success = False
            response.message = f"OWLv2 found no detections for {classes}."
            return response

        counts = [len(self._process_detections(target_class=cls)[1]) for cls in classes]

        max_count = max(counts)
        winners   = [classes[i] for i, cnt in enumerate(counts) if cnt == max_count]

        response.success         = True
        response.compare_classes = classes
        response.compare_counts  = counts

        if len(winners) > 1:
            response.winner_class = "tie"
            response.message = f"Tie between {', '.join(winners)} with {max_count} objects each."
        else:
            response.winner_class = winners[0]
            response.message = f"'{winners[0]}' wins with {max_count} objects."

        self.get_logger().info(response.message)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = VisionAnalyticService()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()