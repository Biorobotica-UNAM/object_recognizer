#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from copy import deepcopy
from functools import partial

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, Trigger

def convolve(img, k):
    """Convolucion 2D (kernel volteado) con padding por replicacion de borde.
    Funciona con kernels no cuadrados y de tamano par (ej. [1 -1])."""
    kh, kw = k.shape
    kf = k[::-1, ::-1]                      # volteo => convolucion verdadera
    pt, pb = (kh - 1) // 2, kh // 2
    pl, pr = (kw - 1) // 2, kw // 2
    h, w = img.shape
    p = np.pad(img, ((pt, pb), (pl, pr)), mode="edge")
    out = np.zeros((h, w), dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            wgt = kf[i, j]
            if wgt != 0.0:
                out += wgt * p[i:i + h, j:j + w]
    return out


def block_kernel(n):
    return np.ones((n, n), dtype=np.float32) / (n * n)


def pascal_row(n):
    """Fila del triangulo de Pascal con n elementos: n=3 -> [1,2,1], n=5 -> [1,4,6,4,1]"""
    row = [1]
    for _ in range(n - 1):
        row = [a + b for a, b in zip([0] + row, row + [0])]
    return row


def binomial_kernel(n):
    r = np.array(pascal_row(n), dtype=np.float32)
    k = np.outer(r, r)
    return k / k.sum()


def _gauss_parts(n):
    """g, g', g'' muestreadas en enteros. sigma^2 = (n-1)/4 (aprox. binomial)."""
    sigma2 = (n - 1) / 4.0
    xs = np.arange(-(n // 2), n // 2 + 1, dtype=np.float32)
    g = np.exp(-(xs ** 2) / (2 * sigma2))
    g /= g.sum()
    d1 = -xs / sigma2 * g
    d1 /= abs(np.sum(xs * d1))              # ganancia unitaria ante una rampa
    d2 = (xs ** 2 / sigma2 ** 2 - 1.0 / sigma2) * g
    return g.astype(np.float32), d1.astype(np.float32), d2.astype(np.float32)


def gauss1_kernels(n):
    """Derivada de primer orden de gaussiana: (Kx, Ky)."""
    g, d1, _ = _gauss_parts(n)
    return np.outer(g, d1), np.outer(d1, g)   # filas = y, columnas = x


def log_kernel(n):
    """Segunda derivada de gaussiana (Laplaciano de gaussiana) de n x n."""
    g, _, d2 = _gauss_parts(n)
    k = np.outer(g, d2) + np.outer(d2, g)
    k -= k.mean()                             # suma cero
    return k.astype(np.float32)


PREWITT_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
PREWITT_Y = PREWITT_X.T.copy()
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
SOBEL_Y = SOBEL_X.T.copy()
LAPLACIAN = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
DIFF_X = np.array([[1, -1]], dtype=np.float32)
DIFF_Y = np.array([[1], [-1]], dtype=np.float32)


def to_display_smooth(x):
    return np.clip(x, 0, 255)


def to_display_edge(x):
    """Magnitud normalizada a 0..255 (para filtros con respuesta con signo)."""
    m = np.abs(x)
    mx = float(m.max())
    return m * (255.0 / mx) if mx > 1e-6 else m


def magnitude(a, b):
    return np.sqrt(a * a + b * b)

class SpatialFiltersNode(Node):
    def __init__(self):
        super().__init__("spatial_filters")

        # Parametros
        self.declare_parameter("camera_topic", "/camera/rgb/image_raw")
        self.declare_parameter("image_path", "")        # opcional (requiere cv2)
        self.declare_parameter("noise_sigma", 20.0)     # desv. estandar del ruido gaussiano
        self.declare_parameter("unsharp_amount", 1.0)   # k del unsharp masking
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("video_mode", False)
        self.declare_parameter("publish_debug", True)
        self.declare_parameter("output_dir", "resultados")

        self.noise_sigma = float(self.get_parameter("noise_sigma").value)
        self.unsharp_amount = float(self.get_parameter("unsharp_amount").value)
        self.output_dir = str(self.get_parameter("output_dir").value)

        # Estado
        self.video_mode = bool(self.get_parameter("video_mode").value)
        self.noise_on = False
        self.current = "original"
        self.last_frame = None
        self.last_image_msg = None
        self.new_frame = False
        self.dirty = True
        self.last_out = None
        self.last_pub_time = 0.0
        self.rng = np.random.default_rng(0)

        self.static_clean = self._load_static_image(self.get_parameter("image_path").value)
        self.static_noisy = self._add_noise(self.static_clean)

        self._kcache = {}
        self.filters = self._build_filters()

        # Servicios: uno por filtro
        for name in self.filters:
            self.create_service(Trigger, f"filters/{name}", partial(self._cb_filter, name=name))
        self.create_service(SetBool, "filters/noise", self._cb_noise)
        self.create_service(Trigger, "filters/save", self._cb_save)

        # Topics
        self.create_subscription(Bool, "filters/video_mode", self._cb_video_mode, 10)
        self.create_subscription(
            Image, self.get_parameter("camera_topic").value,
            self._cb_image, qos_profile_sensor_data)
        self.pub_in = self.create_publisher(Image, "filters/input", 1)
        self.pub_out = self.create_publisher(Image, "filters/output", 1)
        self.pub_original = self.create_publisher(Image, "filters/original", 1)
        self.debug_pubs = {
            name: self.create_publisher(Image, f"filters/debug/{name}", 1)
            for name in self.filters
        }

        rate_hz = float(self.get_parameter("rate_hz").value)
        if not np.isfinite(rate_hz) or rate_hz <= 0:
            raise ValueError("rate_hz debe ser finito y mayor que cero")
        self.create_timer(1.0 / rate_hz, self._tick)
        self.get_logger().info(f"Listo. {len(self.filters)} filtros disponibles: /filters/<nombre>")
        self.get_logger().info("Filtros: " + ", ".join(self.filters))


    def _k(self, key, fn):
        if key not in self._kcache:
            self._kcache[key] = fn()
        return self._kcache[key]

    def _lowpass(self, kind, n):
        if kind == "block":
            return self._k(("block", n), lambda: block_kernel(n))
        return self._k(("bin", n), lambda: binomial_kernel(n))

    def _unsharp(self, img, kind, n):
        """Difumina con block 5x5 (pierde nitidez) y luego aplica unsharp masking.
        mascara = x - lowpass(x);  salida = x + k * mascara"""
        blurred = convolve(img, self._lowpass("block", 5))
        low = convolve(blurred, self._lowpass(kind, n))
        out = blurred + self.unsharp_amount * (blurred - low)
        return to_display_smooth(out)

    def _build_filters(self):
        f = {}
        f["original"] = lambda im: im

        # 2) pasa bajas de bloque y 3) binomiales
        for n in (3, 5, 7, 11):
            f[f"block_{n}"] = lambda im, n=n: to_display_smooth(convolve(im, self._lowpass("block", n)))
            f[f"binomial_{n}"] = lambda im, n=n: to_display_smooth(convolve(im, self._lowpass("bin", n)))

        # 4a) bloque [1 -1]
        f["diff_x"] = lambda im: to_display_edge(convolve(im, DIFF_X))
        f["diff_y"] = lambda im: to_display_edge(convolve(im, DIFF_Y))
        f["diff_mag"] = lambda im: to_display_edge(magnitude(convolve(im, DIFF_X), convolve(im, DIFF_Y)))
        # 4b) Prewitt
        f["prewitt_x"] = lambda im: to_display_edge(convolve(im, PREWITT_X))
        f["prewitt_y"] = lambda im: to_display_edge(convolve(im, PREWITT_Y))
        f["prewitt_mag"] = lambda im: to_display_edge(magnitude(convolve(im, PREWITT_X), convolve(im, PREWITT_Y)))
        # 4c) Sobel
        f["sobel_x"] = lambda im: to_display_edge(convolve(im, SOBEL_X))
        f["sobel_y"] = lambda im: to_display_edge(convolve(im, SOBEL_Y))
        f["sobel_mag"] = lambda im: to_display_edge(magnitude(convolve(im, SOBEL_X), convolve(im, SOBEL_Y)))
        # 4d) primera derivada de gaussiana
        for n in (5, 7, 11):
            def g1(im, n=n):
                kx, ky = self._k(("g1", n), lambda: gauss1_kernels(n))
                return to_display_edge(magnitude(convolve(im, kx), convolve(im, ky)))
            f[f"gauss1_{n}"] = g1

        # 5a) Laplaciano
        f["laplacian"] = lambda im: to_display_edge(convolve(im, LAPLACIAN))
        # 5b) segunda derivada de gaussiana
        for n in (5, 7, 11):
            f[f"gauss2_{n}"] = lambda im, n=n: to_display_edge(
                convolve(im, self._k(("log", n), lambda: log_kernel(n))))

        # 6) unsharp masking (sobre imagen difuminada con block 5x5)
        for kind in ("block", "binomial"):
            for n in (3, 7):
                f[f"unsharp_{kind}_{n}"] = lambda im, kind=kind, n=n: self._unsharp(
                    im, "block" if kind == "block" else "bin", n)
        f["blur_5"] = f["block_5"]
        return f

    def _load_static_image(self, path):
        if path:
            try:
                import cv2  # opcional, solo para leer archivos
                im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if im is not None:
                    return im.astype(np.float32)
                self.get_logger().warn(f"No pude leer {path}, uso imagen sintetica")
            except ImportError:
                self.get_logger().warn("cv2 no disponible, uso imagen sintetica")
        return self._synthetic_image()

    @staticmethod
    def _synthetic_image(h=256, w=256):
        """Imagen de prueba con bordes, circulo, diagonal, detalle fino y degradado."""
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        im = 60 + 40 * xx / w                                  # degradado de fondo
        im[40:110, 30:120] = 200                               # cuadrado
        im[(yy - 160) ** 2 + (xx - 180) ** 2 < 45 ** 2] = 230   # circulo
        im[np.abs((yy - xx) - 40) < 2] = 20                     # diagonal delgada
        im[170:230, 30:90] = np.where(((yy[170:230, 30:90] // 4 + xx[170:230, 30:90] // 4) % 2) == 0, 40, 210)  # tablero
        return im.astype(np.float32)

    def _add_noise(self, img):
        noise = self.rng.normal(0.0, self.noise_sigma, img.shape).astype(np.float32)
        return np.clip(img + noise, 0, 255)

    @staticmethod
    def _decode(msg):
        h, w = msg.height, msg.width
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        if msg.encoding == "mono8":
            return raw.reshape(h, msg.step)[:, :w].astype(np.float32)
        if msg.encoding in ("rgb8", "bgr8"):
            a = raw.reshape(h, msg.step)[:, :w * 3].reshape(h, w, 3).astype(np.float32)
            r, b = (a[..., 0], a[..., 2]) if msg.encoding == "rgb8" else (a[..., 2], a[..., 0])
            return 0.299 * r + 0.587 * a[..., 1] + 0.114 * b
        return None

    def _make_msg(self, img, header=None):
        a = np.clip(img, 0, 255).astype(np.uint8)
        m = Image()
        if header is None:
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = "filters"
        else:
            m.header = deepcopy(header)
        m.height, m.width = a.shape
        m.encoding = "mono8"
        m.is_bigendian = 0
        m.step = a.shape[1]
        m.data = a.tobytes()
        return m


    def _cb_image(self, msg):
        g = self._decode(msg)
        if g is None:
            self.get_logger().warn(f"Encoding no soportado: {msg.encoding}", throttle_duration_sec=5.0)
            return
        self.last_frame = g
        self.last_image_msg = msg
        self.new_frame = True

    def _cb_video_mode(self, msg):
        self.video_mode = bool(msg.data)
        self.new_frame = False
        self.dirty = True
        self.get_logger().info(f"Modo video: {'ON' if self.video_mode else 'OFF (imagen estatica)'}")

    def _cb_filter(self, request, response, name):
        self.current = name
        self.dirty = True
        response.success = True
        response.message = f"Filtro activo: {name}"
        self.get_logger().info(response.message)
        return response

    def _cb_noise(self, request, response):
        self.noise_on = bool(request.data)
        if self.noise_on:
            self.static_noisy = self._add_noise(self.static_clean)
        self.dirty = True
        response.success = True
        response.message = f"Ruido gaussiano (sigma={self.noise_sigma}): {'ON' if self.noise_on else 'OFF'}"
        return response

    def _cb_save(self, request, response):
        if self.last_out is None:
            response.success, response.message = False, "Todavia no hay salida"
            return response
        os.makedirs(self.output_dir, exist_ok=True)
        tag = "ruido" if self.noise_on else "limpia"
        path = os.path.join(self.output_dir, f"{self.current}_{tag}.pgm")
        a = np.clip(self.last_out, 0, 255).astype(np.uint8)
        with open(path, "wb") as fh:                   # PGM binario escrito a mano
            fh.write(f"P5\n{a.shape[1]} {a.shape[0]}\n255\n".encode())
            fh.write(a.tobytes())
        response.success, response.message = True, f"Guardado en {path}"
        return response
    
    def _tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.video_mode:
            if self.last_frame is None or not (self.new_frame or self.dirty):
                return
            src = self.last_frame
            original_msg = self.last_image_msg
            self.new_frame = False
            if self.noise_on:
                src = self._add_noise(src)
        else:
            if not self.dirty and now - self.last_pub_time < 1.0:
                return
            src = self.static_noisy if self.noise_on else self.static_clean
            original_msg = self._make_msg(self.static_clean)
        self.dirty = False
        self.last_pub_time = now

        out = self.filters[self.current](src)
        self.last_out = out
        header = original_msg.header
        output_msg = self._make_msg(out, header)
        self.pub_original.publish(original_msg)
        self.pub_in.publish(self._make_msg(src, header))
        self.pub_out.publish(output_msg)
        if self.get_parameter("publish_debug").value:
            for name, pub in self.debug_pubs.items():
                if pub.get_subscription_count() == 0:
                    continue
                # Todas las salidas usan el mismo frame y la misma muestra de ruido.
                debug_msg = output_msg if name == self.current else self._make_msg(
                    self.filters[name](src), header)
                pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SpatialFiltersNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
