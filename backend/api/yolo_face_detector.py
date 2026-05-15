"""
YOLOv8-face 人脸检测器封装。
使用本地 backend/ultralytics 包（yolov8-face 定制版），避免与系统 ultralytics 冲突。
"""

import os
import sys
import numpy as np
from PIL import Image

# 确保优先使用 backend/ 目录下的 ultralytics（yolov8-face 定制版）
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from ultralytics import YOLO  # noqa: E402  来自 backend/ultralytics/


class YOLOFaceDetector:
    """
    基于 YOLOv8-face 的人脸检测器。

    detect()    → 返回 BGR 图中所有人脸的 xyxy bbox 列表
    crop_face() → 按 bbox 裁剪并加 padding，返回 PIL Image
    """

    def __init__(self, weights_path: str, device: str = 'cpu'):
        """
        Args:
            weights_path: yolov8n-face.pt 等权重文件的绝对或相对路径
            device: 推理设备，'cpu' 或 'cuda'
        """
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"YOLOv8-face 权重文件不存在: {weights_path}")
        self.model = YOLO(weights_path)
        self.device = device
        print(f"[YOLOFaceDetector] 权重已加载: {weights_path}，task={self.model.task}")

    def detect(self, frame_bgr: np.ndarray, conf: float = 0.45) -> list:
        """
        在 BGR 图像上检测所有人脸。

        Args:
            frame_bgr: OpenCV 读取的 BGR numpy 数组，shape (H, W, 3)
            conf: 置信度阈值

        Returns:
            list of dict，每项包含:
                bbox  : [x1, y1, x2, y2]（整数，原图像素坐标）
                conf  : float，检测置信度
        """
        results = self.model.predict(
            source=frame_bgr,
            conf=conf,
            device=self.device,
            verbose=False,
        )
        faces = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = (int(v) for v in xyxy)
                face_conf = float(box.conf[0].cpu().numpy())
                faces.append({
                    'bbox': [x1, y1, x2, y2],
                    'conf': round(face_conf, 4),
                })
        return faces

    def crop_face(
        self,
        frame_bgr: np.ndarray,
        bbox: list,
        pad: float = 0.25,
    ) -> Image.Image:
        """
        按 bbox 裁剪人脸区域，四周扩展 pad 比例的 padding，返回 PIL RGB 图像。

        Args:
            frame_bgr : BGR numpy 数组
            bbox      : [x1, y1, x2, y2]
            pad       : padding 比例（相对于框的宽/高），默认 0.25

        Returns:
            PIL.Image.Image (RGB 模式)
        """
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1

        pad_x = int(bw * pad)
        pad_y = int(bh * pad)

        nx1 = max(0, x1 - pad_x)
        ny1 = max(0, y1 - pad_y)
        nx2 = min(w, x2 + pad_x)
        ny2 = min(h, y2 + pad_y)

        crop_bgr = frame_bgr[ny1:ny2, nx1:nx2]
        # BGR → RGB
        crop_rgb = crop_bgr[:, :, ::-1].copy()
        return Image.fromarray(crop_rgb)
