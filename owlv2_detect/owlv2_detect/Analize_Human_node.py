#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from vision_interfaces.srv import AnalyzePerson, GetDetections
from human_interfaces.srv import DetectPointing

class HumanAnalyticService(Node):
    def __init__(self):
        super().__init__('human_analytic_service')
        
        self.cb_group = ReentrantCallbackGroup()


        #owlv2 service client
        self.owl_client = self.create_client(
            GetDetections, 
            '/vision/owl/owlv2_service', 
            callback_group=self.cb_group
        )

        #pose_detector service client
        self.gesture_client = self.create_client(
            DetectPointing, 
            '/human_detection/detect_wrist', 
            callback_group=self.cb_group
        )
        #service AnalizePerson
        self.srv = self.create_service(
            AnalyzePerson, 
            '/vision/analyze_person', 
            self.handle_analyze_person, 
            callback_group=self.cb_group
        )

        self.get_logger().info(
            "Node HumanAnalyticService ready by /vision/analyze_person"
        )

    async def handle_analyze_person(self, request, response):
        action = request.action.lower().strip()
        response.has_point = False

        if action == "color" or action == "attribute":
            return await self._process_only_color(request, response)

        elif action == "gesture":
            return await self._process_only_gesture(request, response)

        elif action == "combined" or action == "gesture_color":
            return await self._process_combined(request, response)

        else:
            response.success = False
            response.message = f"Action '{action}' invalid. use: color | gesture | combined."
            self.get_logger().error(response.message)
            return response

    def _check_service(self, client, name):

        if not client.service_is_ready():
            client.wait_for_service(timeout_sec=2.0)
        return client.service_is_ready()

    async def _process_only_color(self, request, response):
        color     = request.color.strip().lower()
        attribute = request.attribute.strip().lower()

        if not color or not attribute:
            response.success = False
            response.message = "field 'color'-'attribute' is required for this action."
            return response

        if not self._check_service(self.owl_client, "OWLv2"):
            response.success = False
            response.message = "Service 'OWLv2' is offline."
            return response

        owl_prompt = f"{color} {attribute}"
        owl_req = GetDetections.Request()
        owl_req.queries = [owl_prompt]

        self.get_logger().info(f"Calling OWLv2 with prompt: '{owl_prompt}'")
        owl_resp = await self.owl_client.call_async(owl_req)

        response.detections = []

        if owl_resp.success and len(owl_resp.detections) > 0:
            best = owl_resp.detections[0]
            response.success    = True
            response.message    = f"person_found_with_{attribute}_{color}"
            response.detections = owl_resp.detections
            self.get_logger().info(f"Exit, person found with {attribute} {color} (conf={best.confidence:.2f})")
        else:
            response.success = False
            response.message = f"do not found person with {attribute} {color}."
            self.get_logger().warn(response.message)
            
        return response

    async def _process_only_gesture(self, request, response):
        gesture = request.gesture_type.strip().lower()
        arm     = request.arm_side.strip().lower() if request.arm_side else "any"

        if not gesture:
            response.success = False
            response.message = "field 'gesture_type' is required for this action."
            return response

        if not self._check_service(self.gesture_client, "pose_detector"):
            response.success = False
            response.message = "Service 'pose_detector' is offline."
            return response

        gesture_req = DetectPointing.Request()
        gesture_req.action_type = gesture
        gesture_req.arm_side    = arm

        self.get_logger().info(f"Calling pose_detector to evaluate gesture: '{gesture}'")
        gesture_resp = await self.gesture_client.call_async(gesture_req)
        response.detections = []

        if gesture_resp.human_detected:
            response.success = True
            response.message = f"person_found_doing_{gesture_resp.direction}"
            response.has_point = True
            response.point = gesture_resp.point
            self.get_logger().info(f"Exit, gesture detected in physical direction: {gesture_resp.direction}")
        else:
            response.success = False
            response.message = f"do not found person doing gesture '{gesture}'."
            self.get_logger().warn(response.message)

        return response

    async def _process_combined(self, request, response):
        gesture   = request.gesture_type.strip().lower()
        color     = request.color.strip().lower()
        attribute = request.attribute.strip().lower()
        arm       = request.arm_side.strip().lower() if request.arm_side else "any"

        self.get_logger().info(f"Evaluating combination -> Gesture: {gesture} | Clothing: {color} {attribute}")

        if not self._check_service(self.gesture_client, "pose_detector"):
            response.success    = False
            response.detections = []
            response.message    = "Service 'pose_detector' is offline."
            return response

        gesture_req = DetectPointing.Request()
        gesture_req.action_type = gesture
        gesture_req.arm_side    = arm
        
        gesture_resp = await self.gesture_client.call_async(gesture_req)
        response.detections = []

        if not gesture_resp.human_detected:
            response.success = False
            response.message = f"Filter failed: No one is performing the gesture '{gesture}'."
            self.get_logger().warn(response.message)
            return response

        response.has_point = True
        response.point = gesture_resp.point

        self.get_logger().info(f"Gesture '{gesture}' validated in {gesture_resp.direction}. Running clothing verification...")
        if not self._check_service(self.owl_client, "OWLv2"):
            response.success = False
            response.message = "Service 'OWLv2' is offline."
            return response

        owl_prompt = f"{color} {attribute}"
        owl_req = GetDetections.Request()
        owl_req.queries = [owl_prompt]

        owl_resp = await self.owl_client.call_async(owl_req)

        if owl_resp.success and len(owl_resp.detections) > 0:
            best = owl_resp.detections[0]
            response.success    = True
            response.message    = f"person_found_doing_{gesture_resp.direction}_with_{attribute}_{color}"
            response.detections = owl_resp.detections
            self.get_logger().info(f"Exit, person found doing {gesture_resp.direction} with {attribute} {color} (OWL Conf: {best.confidence:.2f})")
        else:
            response.success = False
            response.message = f"Gesture correct, but the person is not wearing a {attribute} {color}."
            self.get_logger().warn(response.message)

        return response


def main(args=None):
    rclpy.init(args=args)
    node = HumanAnalyticService()
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
