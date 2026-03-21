#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, LivelinessPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import PointCloud2
import ros2_numpy 
from vision_interfaces.msg import YoloDetections
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from std_srvs.srv import Trigger
import numpy as np
import threading

class DetectSeatsNode(Node):
    def __init__(self):
        super().__init__('detect_seats_node')

        self.declare_parameter("pointcloud_topic", "/kinect/points")
        pointcloud_topic = self.get_parameter("pointcloud_topic").value

        # QoS optimizada para sensores de alto ancho de banda
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=1
        )
        
        self.group = ReentrantCallbackGroup()
        
        # Suscripciones
        self.create_subscription(YoloDetections, 'detections_list', self._det_cb, 1, callback_group=self.group)  
        self.create_subscription(PointCloud2, pointcloud_topic, self._pc_cb, qos_profile, callback_group=self.group)
        
        self.yolo_client = self.create_client(Trigger, 'detection', callback_group=self.group)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.srv = self.create_service(Trigger, '/check_seats', self._handle_service, callback_group=self.group)

        # Variables de estado
        self._last_dets = None
        self._last_pc = None
        self._det_event = threading.Event() # Evento para evitar el while time.sleep
        
        self.active_tfs = {}
        # Bajamos la frecuencia de TF si no es crítica para ahorrar CPU
        self.tf_timer = self.create_timer(0.1, self._manage_tfs, callback_group=self.group)
        
        self.get_logger().info("DETECT_SEATS OPTIMIZED READY...")

    def _det_cb(self, msg: YoloDetections):
        self._last_dets = msg
        self._det_event.set() # Avisa que llegaron nuevas detecciones

    def _pc_cb(self, msg: PointCloud2):
        self._last_pc = msg

    def _manage_tfs(self):
        if not self.active_tfs:
            return
        now = self.get_clock().now().to_msg()
        for t_msg in self.active_tfs.values():
            t_msg.header.stamp = now
            self.tf_broadcaster.sendTransform(t_msg)

    async def _handle_service(self, req, res):
        if not self.yolo_client.wait_for_service(timeout_sec=1.0):
            res.success = False
            res.message = "YOLO service unavailable"
            return res
        
        self._det_event.clear()
        self.get_logger().info("Checking seats...")
        
        # Llamada asíncrona sin bloquear el executor
        future = self.yolo_client.call_async(Trigger.Request())
        
        # Esperar hasta 3 segundos por la respuesta O por el evento de detección
        finished = self._det_event.wait(timeout=3.0)

        if not finished or self._last_pc is None:
            res.success = False
            res.message = "Timeout or missing data"
            return res

        # Cache de datos para evitar que cambien durante el procesamiento
        current_dets = self._last_dets.detections
        current_pc = self._last_pc 

        persons = [d for d in current_dets if d.class_name == 'person']
        seats = [d for d in current_dets if d.class_name in ['chair', 'couch', 'sofa']]

        best_seat_pose = None
        min_dist = float('inf')

        # Convertimos la nube una sola vez por servicio, no por cada bbox
        try:
            pc_data = ros2_numpy.numpify(current_pc)
        except Exception as e:
            res.success = False
            res.message = f"Pointcloud error: {e}"
            return res

        for s in seats:
            s_box = [s.x_min, s.y_min, s.x_max, s.y_max]
            # Vectorización implícita: calculamos ocupación
            is_occupied = any(self._calculate_overlap([p.x_min, p.y_min, p.x_max, p.y_max], s_box) > 0.30 for p in persons)
            
            if not is_occupied:
                pos = self._get_3d_pose_optimized(pc_data, s_box)
                if pos:
                    dist = np.linalg.norm(pos)
                    if dist < min_dist:
                        min_dist = dist
                        best_seat_pose = pos
        
        if best_seat_pose:
            t = TransformStamped()
            t.header.frame_id = current_pc.header.frame_id
            t.child_frame_id = "free_seat"
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = best_seat_pose
            t.transform.rotation.w = 1.0
            
            self.active_tfs["free_seat"] = t
            res.success = True
            res.message = f"Found at {best_seat_pose}"
        else:
            res.success = False
            res.message = "No free seats"
            
        return res

    def _get_3d_pose_optimized(self, pc_data, bbox):
        try:
            h, w = self._last_pc.height, self._last_pc.width

            # SOPORTE DE FORMATOS
            if isinstance(pc_data, dict):
                keys = pc_data.keys()

                if all(k in keys for k in ['x', 'y', 'z']):
                    img_3d = np.stack([pc_data['x'], pc_data['y'], pc_data['z']], axis=-1)

                elif 'xyz' in keys:
                    img_3d = pc_data['xyz']

                else:
                    self.get_logger().error(f"Dict sin campos válidos: {keys}")
                    return None

            elif isinstance(pc_data, np.ndarray):

                if pc_data.ndim == 3 and pc_data.shape[2] == 3:
                    img_3d = pc_data

                elif pc_data.ndim == 2 and pc_data.shape[1] == 3:
                    img_3d = pc_data.reshape(h, w, 3)

                elif pc_data.dtype.names:
                    fields = pc_data.dtype.names

                    if all(k in fields for k in ['x', 'y', 'z']):
                        img_3d = np.stack([pc_data['x'], pc_data['y'], pc_data['z']], axis=-1)

                    elif 'xyz' in fields:
                        img_3d = pc_data['xyz']

                    else:
                        self.get_logger().error(f"Campos no soportados: {fields}")
                        return None
                else:
                    self.get_logger().error(f"Formato ndarray raro: {pc_data.shape}")
                    return None

            else:
                self.get_logger().error(f"Tipo desconocido: {type(pc_data)}")
                return None

            # Asegurar shape
            if img_3d.shape[0] != h or img_3d.shape[1] != w:
                img_3d = img_3d.reshape(h, w, 3)

            # BBOX
            x1, y1, x2, y2 = map(int, bbox)
            x1 = np.clip(x1, 0, w - 1)
            x2 = np.clip(x2, 0, w - 1)
            y1 = np.clip(y1, 0, h - 1)
            y2 = np.clip(y2, 0, h - 1)

            if x2 <= x1 or y2 <= y1:
                return None

            roi = img_3d[y1:y2, x1:x2]

            # puntos válidos
            zs = roi[:, :, 2]
            mask = np.isfinite(zs) & (zs > 0)

            if not np.any(mask):
                return None

            # mediana
            mx = np.median(roi[:, :, 0][mask])
            my = np.median(roi[:, :, 1][mask])
            mz = np.median(zs[mask])

            # filtro
            if mz > 5.0 or mz < 0.3:
                return None

            return (float(mx), float(my), float(mz))

        except Exception as e:
            self.get_logger().error(f"Fallo total en 3D: {str(e)}")
            return None

    def _calculate_overlap(self, b_per, b_seat):
        # Lógica de intersección sobre unión simplificada
        dx = min(b_per[2], b_seat[2]) - max(b_per[0], b_seat[0])
        dy = min(b_per[3], b_seat[3]) - max(b_per[1], b_seat[1])
        if dx > 0 and dy > 0:
            area_overlap = dx * dy
            area_seat = (b_seat[2] - b_seat[0]) * (b_seat[3] - b_seat[1])
            return area_overlap / area_seat
        return 0.0

def main():
    rclpy.init()
    node = DetectSeatsNode()
    # MultiThreadedExecutor es vital aquí para procesar TF y Servicio a la vez
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()