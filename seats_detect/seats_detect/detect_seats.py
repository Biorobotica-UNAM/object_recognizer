#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import PointCloud2
import ros2_numpy 
from vision_interfaces.msg import YoloDetections
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from std_srvs.srv import Trigger
import numpy as np
import time

class DetectSeatsNode(Node):
    def __init__(self):
        super().__init__('detect_seats_node')
        
        # Grupo de callbacks para permitir ejecución paralela (se agregan las subscripciones,servicios y clientes al mismo grupo para que puedan ejecutarse simultáneamente)
        self.group = ReentrantCallbackGroup()
        self.create_subscription(YoloDetections, '/yolo/detections_list', self._det_cb, 10, callback_group=self.group)  
        self.create_subscription(PointCloud2, '/kinect/points', self._pc_cb, 10, callback_group=self.group)
        self.yolo_client = self.create_client(Trigger, '/yolo/Detection', callback_group=self.group)
        
        self.tf_broadcaster = TransformBroadcaster(self)
        self.srv = self.create_service(Trigger, '/check_seats', self._handle_service, callback_group=self.group)

        self._last_dets = None
        self._last_pc = None
        self.active_tfs = {}
        self.tf_timer = self.create_timer(0.1, self._manage_tfs, callback_group=self.group)
        
        self.get_logger().info("DETECT_SEATS READY...")

    def _det_cb(self, msg: YoloDetections):
        self._last_dets = msg

    def _pc_cb(self, msg: PointCloud2):
        self._last_pc = msg

    def _manage_tfs(self):
        for frame_id, t_msg in self.active_tfs.items():
            t_msg.header.stamp = self.get_clock().now().to_msg()
            self.tf_broadcaster.sendTransform(t_msg)

    def _handle_service(self, req, res):
        # se manda llamar al servicio de YOLO para obtener las detecciones más recientes
        if not self.yolo_client.wait_for_service(timeout_sec=2.0):
            res.success = False
            res.message = "ERROR: YOLO service not available."
            return res
        
        self._last_dets = None
     
        self.get_logger().info("Request received: Checking for free seats...")
        yolo_req = Trigger.Request()
        yolo_future = self.yolo_client.call_async(yolo_req)
        
        start_t = time.time()
        while self._last_dets is None and (time.time() - start_t) < 3.0:
            time.sleep(0.1)

        if self._last_dets is None or self._last_pc is None:
            res.success = False
            res.message = "ERROR: No detections or point cloud received in time."
            return res

        persons = [d for d in self._last_dets.detections if d.class_name == 'person']
        seats = [d for d in self._last_dets.detections if d.class_name in ['chair', 'couch']]

        best_seat = None
        min_dist = float('inf')

        for s in seats:
            s_box = [s.x_min, s.y_min, s.x_max, s.y_max]
            is_occupied = any(self._calculate_overlap([p.x_min, p.y_min, p.x_max, p.y_max], s_box) > 0.30 for p in persons)
            
            if not is_occupied:
                point_3d = self._get_3d_pose_from_bbox(s.x_min, s.y_min, s.x_max, s.y_max)
                if point_3d:
                    dist = np.sqrt(point_3d[0]**2 + point_3d[1]**2 + point_3d[2]**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_seat = point_3d
        
        self.active_tfs.clear()

        if best_seat:
            frame_id = "free_seat"
            t = TransformStamped()
            t.header.frame_id = self._last_pc.header.frame_id
            t.child_frame_id = frame_id
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = best_seat
            t.transform.rotation.w = 1.0
            
            self.active_tfs[frame_id] = t
            formatted_coords = f"(x: {best_seat[0]:.2f}, y: {best_seat[1]:.2f}, z: {best_seat[2]:.2f})"
            res.success = True
            res.message = f"FREE SEATS IN: {formatted_coords}"
            self.get_logger().info(res.message)
        else:
            res.success = False
            res.message = "No free seats found."
            
        return res
    
    def _calculate_overlap(self, b_per, b_seat):
        ix1, iy1 = max(b_per[0], b_seat[0]), max(b_per[1], b_seat[1])
        ix2, iy2 = min(b_per[2], b_seat[2]), min(b_per[3], b_seat[3])
        if ix2 < ix1 or iy2 < iy1: return 0.0
        return ((ix2 - ix1) * (iy2 - iy1)) / ((b_seat[2] - b_seat[0]) * (b_seat[3] - b_seat[1]))

    def _get_3d_pose_from_bbox(self, x1, y1, x2, y2):
        try:
            pc_data = ros2_numpy.numpify(self._last_pc)
            h, w = pc_data.shape
            x1_idx, x2_idx = max(0, int(x1)), min(w, int(x2))
            y1_idx, y2_idx = max(0, int(y1)), min(h, int(y2))
            roi_x = pc_data['x'][y1_idx:y2_idx, x1_idx:x2_idx]
            roi_y = pc_data['y'][y1_idx:y2_idx, x1_idx:x2_idx]
            roi_z = pc_data['z'][y1_idx:y2_idx, x1_idx:x2_idx]
            mx, my, mz = np.nanmean(roi_x), np.nanmean(roi_y), np.nanmean(roi_z)
            return (float(mx), float(my), float(mz)) if not np.isnan(mx) else None
        except: return None

def main():
    rclpy.init()
    node = DetectSeatsNode()
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