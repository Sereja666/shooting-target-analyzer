"""Разбор кадров одной живой сессии. Окно OpenCV сюда не входит."""

from __future__ import annotations

import struct
import threading
import time
from math import hypot
from random import uniform

import cv2
import numpy as np

from shottracker.demo import DemoWorld
from shottracker.detector import HoleDetector, blob_limits
from shottracker.geometry import Calibration, find_circle, work_view
from shottracker.targets import TARGETS, score_at

DUP = 0.012
CROP_MAGIC = b"STC1"
CROP_HEADER = 4 + 5 * 4


class LiveAnalyzer:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.detector = HoleDetector()
        self.calib: Calibration | None = None
        self.analysis = False
        self.mode = "browser"
        self.target_key = "pistol"
        self.size_scale = 1.0
        self.world: DemoWorld | None = None
        self.cap: cv2.VideoCapture | None = None
        self.points: list[tuple[float, float]] = []
        self.t0: float | None = None
        self._next_shot: float | None = None
        self.stopped = False
        self.last_status = ""

    def configure(
        self,
        target_key: str,
        sensitivity: int,
        size_scale: float,
        points: list[tuple[float, float]],
        calib: Calibration | None,
    ) -> None:
        with self.lock:
            self.target_key = target_key if target_key in TARGETS else "pistol"
            self.detector.sensitivity = int(np.clip(sensitivity, 8, 80))
            self.size_scale = float(np.clip(size_scale, 0.4, 2.5))
            self.points = list(points)
            self.calib = calib

    def set_target(self, key: str) -> None:
        with self.lock:
            if key not in TARGETS:
                return
            self.target_key = key
            if self.world is not None:
                self.world.set_target(key)
            self.detector.reset()
            self._next_shot = None

    def set_sensitivity(self, value: int) -> None:
        with self.lock:
            self.detector.sensitivity = int(np.clip(value, 8, 80))

    def set_size(self, value: float) -> None:
        with self.lock:
            self.size_scale = float(np.clip(value, 0.4, 2.5))

    def set_calib(self, calib: Calibration) -> None:
        with self.lock:
            self.calib = calib
            self.detector.reset()
            self._next_shot = None

    def start(self) -> str | None:
        with self.lock:
            if self.calib is None:
                return "Сначала отметьте центр и край зоны 1"
            self.analysis = True
            if self.t0 is None:
                self.t0 = time.monotonic()
            return None

    def stop(self) -> None:
        with self.lock:
            self.analysis = False

    def reset_reference(self) -> None:
        with self.lock:
            self.detector.reset()
            self._next_shot = None

    def undo_point(self, norm_x: float, norm_y: float) -> None:
        with self.lock:
            for index in range(len(self.points) - 1, -1, -1):
                x, y = self.points[index]
                if hypot(x - norm_x, y - norm_y) < 1e-6:
                    self.points.pop(index)
                    return

    def start_demo(self) -> None:
        with self.lock:
            self._release_cap()
            self.world = DemoWorld(target_key=self.target_key)
            self.mode = "demo"
            self.calib = Calibration(
                "raw", self.world.center, self.world.outer_r, None, (self.world.w, self.world.h),
            )
            self.detector.reset()
            self.analysis = False
            self._next_shot = None
            self.last_status = ""

    def start_browser(self, calib: Calibration | None) -> None:
        with self.lock:
            self._release_cap()
            self.world = None
            self.mode = "browser"
            self.calib = calib
            self.analysis = False
            self.detector.reset()
            self._next_shot = None
            self.last_status = ""

    def start_ip(self, url: str, calib: Calibration | None) -> str | None:
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            cap.release()
            return "Камера по этому адресу не открылась"
        with self.lock:
            self._release_cap()
            self.world = None
            self.cap = cap
            self.mode = "ip"
            self.calib = calib
            self.analysis = False
            self.detector.reset()
            self._next_shot = None
            self.last_status = ""
        return None

    def close(self) -> None:
        with self.lock:
            self.stopped = True
            self.analysis = False
            self.mode = "closed"
            self._release_cap()

    def _release_cap(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def grab(self) -> tuple[bytes, dict] | None:
        mode = self.mode
        frame = None
        if mode == "ip":
            cap = self.cap
            if cap is None:
                return None
            ok, frame = cap.read()
            if not ok or frame is None:
                return None
            frame = downscale(frame)
        with self.lock:
            if self.stopped or self.mode != mode:
                return None
            if mode == "demo":
                if self.world is None:
                    return None
                self._maybe_hole()
                frame = self.world.frame()
            if frame is None:
                return None
            result = self._step(frame)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
        if not ok:
            return None
        return buf.tobytes(), result

    def step_jpeg(self, data: bytes) -> dict:
        crop = None
        if data[:4] == CROP_MAGIC and len(data) > CROP_HEADER:
            crop = struct.unpack("<5f", data[4:CROP_HEADER])
            data = data[CROP_HEADER:]
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return {"status": "bad-frame", "hits": []}
        if crop is None:
            frame = downscale(frame)
        with self.lock:
            if self.mode != "browser":
                return {"status": "paused", "hits": []}
            calib = self.calib
            if crop is not None and calib is not None:
                calib = crop_calib(calib, crop, frame.shape[1], frame.shape[0])
            return self._step(frame, calib)

    def manual(self, norm_x: float, norm_y: float) -> dict | None:
        with self.lock:
            if hypot(norm_x, norm_y) > 1.15:
                return None
            if self._duplicate(norm_x, norm_y):
                return None
            return self._pack(norm_x, norm_y, "manual")

    def auto_from_frame(self, frame: np.ndarray) -> Calibration | None:
        found = find_circle(frame)
        if found is None:
            return None
        x, y, radius = found
        height, width = frame.shape[:2]
        calib = Calibration("raw", (x, y), radius, None, (width, height))
        self.set_calib(calib)
        return calib

    def _maybe_hole(self) -> None:
        if not self.analysis or self.world is None or self.detector.reference is None:
            return
        now = time.monotonic()
        if self._next_shot is None:
            self._next_shot = now + 1.3
        elif now >= self._next_shot:
            self.world.add_random_hole()
            self._next_shot = now + uniform(3.2, 5.6)

    def _step(self, frame: np.ndarray, calib: Calibration | None = None) -> dict:
        calib = calib or self.calib
        if calib is None:
            return {"status": "need-calib", "hits": []}
        if not self.analysis:
            return {"status": "paused", "hits": []}
        try:
            view = work_view(frame, calib)
        except (ValueError, cv2.error):
            return {"status": "bad-calib", "hits": []}
        min_area, max_area = blob_limits(view.outer_r, self.size_scale)
        self.detector.min_area = min_area
        self.detector.max_area = max_area
        result = self.detector.process(view.image)
        found = []
        if result.status == "ok" and view.outer_r > 1:
            for blob in result.new_holes:
                norm_x = (blob.x - view.cx) / view.outer_r
                norm_y = (blob.y - view.cy) / view.outer_r
                if hypot(norm_x, norm_y) > 1.12:
                    continue
                if self._duplicate(norm_x, norm_y):
                    continue
                found.append(self._pack(norm_x, norm_y, "camera"))
        return {"status": result.status, "hits": found}

    def _duplicate(self, norm_x: float, norm_y: float) -> bool:
        return any(hypot(norm_x - x, norm_y - y) < DUP for x, y in self.points)

    def _pack(self, norm_x: float, norm_y: float, source: str) -> dict:
        self.points.append((norm_x, norm_y))
        score = score_at(norm_x, norm_y, TARGETS[self.target_key])
        offset = 0 if self.t0 is None else int((time.monotonic() - self.t0) * 1000)
        return {
            "norm_x": float(norm_x),
            "norm_y": float(norm_y),
            "score": str(score),
            "source": source,
            "offset_ms": max(0, offset),
        }


def crop_calib(calib: Calibration, crop: tuple[float, ...], width: int, height: int) -> Calibration:
    """Калибровка полного кадра, пересчитанная в квадрат, который вырезал браузер.

    crop — x0, y0, сторона квадрата и размер полного кадра, всё в пикселях камеры.
    """
    x0, y0, side, full_w, full_h = crop
    full = calib.scaled_to(int(round(full_w)), int(round(full_h)))
    k = width / max(side, 1.0)
    return Calibration(
        "raw",
        ((full.center[0] - x0) * k, (full.center[1] - y0) * k),
        full.outer_r * k,
        None,
        (width, height),
    )


def downscale(frame: np.ndarray, limit: int = 1280) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= limit:
        return frame
    scale = limit / float(width)
    size = (limit, max(1, int(round(height * scale))))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
