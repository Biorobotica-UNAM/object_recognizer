#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from vision_interfaces.msg import YoloDetections, YoloDetection
from vision_interfaces.srv import GetAnalytics

class VisionAnalyticService(Node):
    def __init__(self):
        super().__init__('vision_analytic_service')
        

        self.declare_parameter("iou_threshold", 0.3)
        self.iou_threshold = self.get_parameter("iou_threshold").value

        # Estado interno para tracking y conteo
        self.last_detections = []
        self.cumulative_count = 0
        self.object_history = {} # {id: bounding_box}
        self.next_id = 0

        self.sub = self.create_subscription(
            YoloDetections, 
            "/vision/detections", 
            self.cache_callback, 
            10
        )
        
        self.srv = self.create_service(
            GetAnalytics, 
            'vision/get_analytics', 
            self.handle_analytics
        )

        self.get_logger().info("Nodo de Analisis listo...")

    def cache_callback(self, msg):
        self.last_detections = msg.detections

    def calculate_iou(self, boxA, boxB):
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(areaA + areaB - interArea) if (areaA + areaB - interArea) > 0 else 0

    def find_match(self, new_box):
        best_id, max_iou = None, self.iou_threshold
        for obj_id, old_box in self.object_history.items():
            iou = self.calculate_iou(new_box, old_box)
            if iou > max_iou:
                max_iou, best_id = iou, obj_id
        return best_id

    def handle_analytics(self, request, response):
        if request.reset_count:
            self.cumulative_count, self.object_history, self.next_id = 0, {}, 0
            self.get_logger().info("Contador reseteado.")

        if not self.last_detections:
            response.success = False
            response.message = "No hay detecciones recientes en memoria."
            return response

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
            
            obj_data = {'msg': new_det, 'area': area, 'raw_label': det.class_name}
            all_processed.append(obj_data)


            if not request.target_class or request.target_class.lower() in det.class_name.lower():
                target_only.append(obj_data)

        # Ordena por tamaño (Grande a Pequeño)
        target_only.sort(key=lambda x: x['area'], reverse=True)
        
        #  Aplica filtros de modo (big, small, mixed, all)
        mode = request.filter_mode.lower()
        limit = request.count_limit if request.count_limit > 0 else 1
        final_selection = []

        if mode == "big":
            final_selection = target_only[:limit]
        elif mode == "small":
            final_selection = target_only[::-1][:limit]
        elif mode == "mixed" and len(target_only) >= 2:
            final_selection = [target_only[0], target_only[-1]]
        else: 
            final_selection = target_only

        response.success = True
        response.total_cumulative = self.cumulative_count
        response.current_count = len(self.last_detections)
        response.class_count = len(target_only)
        response.detections = [x['msg'] for x in final_selection]
        response.message = f"find {len(target_only)} de '{request.target_class}'. Mode: {mode}."

        self.get_logger().info(response.message)
        return response

def main(args=None):
    rclpy.init(args=args)
    node = VisionAnalyticService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
