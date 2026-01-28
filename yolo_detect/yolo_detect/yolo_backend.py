from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import cv2

@dataclass
class Detection:
    class_name: str
    conf: float
    xyxy: Tuple[int, int, int, int]

class YoloBackend:
    def __init__(self, model_path: str, device: str = "cuda:0", conf_th: float = 0.5):
        self.model = YOLO(model_path)
        self.device = device
        self.conf_th = float(conf_th)

    def infer(self, bgr: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            source=bgr, device=self.device, conf=self.conf_th, verbose=False
        )
        if not results:
            return []
        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return []

        names = r0.names
        xyxy = r0.boxes.xyxy.cpu().numpy()
        cls = r0.boxes.cls.cpu().numpy().astype(int)
        conf = r0.boxes.conf.cpu().numpy()

        h, w = bgr.shape[:2]
        dets: List[Detection] = []
        for (x1, y1, x2, y2), cid, c in zip(xyxy, cls, conf):
            x1i = int(np.clip(round(x1), 0, w - 1))
            y1i = int(np.clip(round(y1), 0, h - 1))
            x2i = int(np.clip(round(x2), 0, w - 1))
            y2i = int(np.clip(round(y2), 0, h - 1))
            if x2i <= x1i or y2i <= y1i:
                continue
            dets.append(Detection(str(names.get(int(cid), str(cid))), float(c), (x1i, y1i, x2i, y2i)))
        return dets
    
    def draw_detections(self, bgr: np.ndarray, dets: List[Detection]) -> np.ndarray:
        drawn = bgr.copy()
        for det in dets:
            x1, y1, x2, y2 = det.xyxy
            cv2.rectangle(drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det.class_name}: {det.conf:.2f}"
            cv2.putText(drawn, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return drawn

    @staticmethod
    def pick_best_by_class(dets: List[Detection], wanted: str) -> Optional[Detection]:
        wanted = wanted.strip().lower()
        cand = [d for d in dets if d.class_name.strip().lower() == wanted]
        if not cand:
            return None
        return max(cand, key=lambda d: d.conf)
