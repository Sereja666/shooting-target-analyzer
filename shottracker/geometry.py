"""Калибровка: прямая проекция или четыре угла листа."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

WORK = 640
CROP_FACTOR = 2.5


@dataclass
class Calibration:
    space: str
    center: tuple[float, float]
    outer_r: float
    corners: list[tuple[float, float]] | None = None
    frame_size: tuple[int, int] | None = None

    def scaled_to(self, width: int, height: int) -> Calibration:
        if self.frame_size is None or self.frame_size == (width, height):
            return self
        src_w, src_h = self.frame_size
        sx = width / src_w
        sy = height / src_h
        corners = None
        if self.corners:
            corners = [(x * sx, y * sy) for x, y in self.corners]
        if self.space == "warp":
            return Calibration("warp", self.center, self.outer_r, corners, (width, height))
        scale = (sx + sy) / 2.0
        return Calibration(
            "raw",
            (self.center[0] * sx, self.center[1] * sy),
            self.outer_r * scale,
            corners,
            (width, height),
        )


@dataclass
class WorkView:
    image: np.ndarray
    cx: float
    cy: float
    outer_r: float


def translate(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        image, matrix, (width, height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )


def warp_frame(frame: np.ndarray, corners: list[tuple[float, float]], size: int = WORK) -> np.ndarray:
    src = np.array(corners, np.float32)
    dst = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, matrix, (size, size), flags=cv2.INTER_LINEAR)


def quad_ok(points: list[tuple[float, float]]) -> bool:
    if len(points) != 4:
        return False
    contour = np.array(points, np.float32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour):
        return False
    return cv2.contourArea(contour) > 2000


def crop_square(image: np.ndarray, cx: float, cy: float, side: int) -> tuple[np.ndarray, int, int]:
    side = int(side)
    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))
    x2 = x1 + side
    y2 = y1 + side
    height, width = image.shape[:2]
    pad_l, pad_t = max(0, -x1), max(0, -y1)
    pad_r, pad_b = max(0, x2 - width), max(0, y2 - height)
    src = image
    if pad_l or pad_t or pad_r or pad_b:
        src = cv2.copyMakeBorder(image, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
    crop = src[y1 + pad_t:y2 + pad_t, x1 + pad_l:x2 + pad_l]
    if crop.shape[0] != side or crop.shape[1] != side:
        raise ValueError(f"crop {crop.shape} != {side}")
    return crop, x1, y1


def work_view(frame: np.ndarray, calib: Calibration) -> WorkView:
    height, width = frame.shape[:2]
    calib = calib.scaled_to(width, height)
    if calib.space == "warp":
        if not calib.corners or len(calib.corners) != 4:
            raise ValueError("для перспективы нужны 4 угла")
        image = warp_frame(frame, calib.corners, WORK)
        return WorkView(image, calib.center[0], calib.center[1], calib.outer_r)
    side = max(32, int(round(calib.outer_r * CROP_FACTOR)))
    crop, x1, y1 = crop_square(frame, calib.center[0], calib.center[1], side)
    interp = cv2.INTER_AREA if crop.shape[1] > WORK else cv2.INTER_LINEAR
    image = cv2.resize(crop, (WORK, WORK), interpolation=interp)
    scale = WORK / float(crop.shape[1])
    return WorkView(
        image,
        (calib.center[0] - x1) * scale,
        (calib.center[1] - y1) * scale,
        calib.outer_r * scale,
    )


def find_circle(image: np.ndarray) -> tuple[float, float, float] | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    height, width = gray.shape
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(40, height // 3),
        param1=120,
        param2=42,
        minRadius=max(12, int(min(width, height) * 0.12)),
        maxRadius=int(min(width, height) * 0.49),
    )
    if circles is None:
        return None
    limit = 0.28 * min(width, height)
    mid_x, mid_y = width / 2.0, height / 2.0
    ranked = []
    for x, y, radius in circles[0]:
        if ((x - mid_x) ** 2 + (y - mid_y) ** 2) ** 0.5 <= limit:
            ranked.append((float(radius), float(x), float(y)))
    if not ranked:
        ranked = [(float(radius), float(x), float(y)) for x, y, radius in circles[0]]
    radius, x, y = max(ranked)
    return x, y, radius


def save_calibration(path: Path, calib: Calibration, target_key: str) -> None:
    payload = {
        "space": calib.space,
        "center": [calib.center[0], calib.center[1]],
        "outer_r": calib.outer_r,
        "corners": calib.corners,
        "frame_size": list(calib.frame_size) if calib.frame_size else None,
        "target": target_key,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_calibration(path: Path) -> tuple[Calibration, str] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        corners = data.get("corners")
        if corners:
            corners = [(float(point[0]), float(point[1])) for point in corners]
        frame_size = data.get("frame_size")
        calib = Calibration(
            space=str(data["space"]),
            center=(float(data["center"][0]), float(data["center"][1])),
            outer_r=float(data["outer_r"]),
            corners=corners,
            frame_size=(int(frame_size[0]), int(frame_size[1])) if frame_size else None,
        )
        if calib.space not in ("raw", "warp") or calib.outer_r < 10:
            return None
        return calib, str(data.get("target") or "pistol")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
