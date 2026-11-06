"""Синтетическая мишень, чтобы проверить детектор без тира и камеры."""

from __future__ import annotations

import cv2
import numpy as np

from shottracker.targets import TARGETS, paint_rings


def render_sheet(width: int, height: int, target_key: str, center: tuple[float, float], outer_r: float) -> np.ndarray:
    image = np.full((height, width, 3), (34, 32, 30), np.uint8)
    cx, cy = int(round(center[0])), int(round(center[1]))
    margin = int(outer_r * 0.18)
    x1 = max(0, cx - int(outer_r) - margin)
    y1 = max(0, cy - int(outer_r) - margin)
    x2 = min(width - 1, cx + int(outer_r) + margin)
    y2 = min(height - 1, cy + int(outer_r) + margin)
    cv2.rectangle(image, (x1, y1), (x2, y2), (214, 226, 236), -1)
    cv2.rectangle(image, (x1, y1), (x2, y2), (80, 80, 80), 2)
    paint_rings(image, center[0], center[1], outer_r, TARGETS[target_key])
    return image


class DemoWorld:
    def __init__(self, width: int = 960, height: int = 720, seed: int = 1, noise: float = 1.8, target_key: str = "pistol"):
        self.w = width
        self.h = height
        self.center = (width / 2.0, height / 2.0)
        self.outer_r = min(width, height) * 0.40
        self.noise = noise
        self.target_key = target_key
        self.rng = np.random.default_rng(seed)
        self.holes: list[tuple[int, int, int, tuple[int, int, int]]] = []
        self._plain: np.ndarray | None = None

    @property
    def plain(self) -> np.ndarray:
        if self._plain is None:
            self._plain = render_sheet(self.w, self.h, self.target_key, self.center, self.outer_r)
        return self._plain

    def set_target(self, key: str) -> None:
        self.target_key = key
        self._plain = None
        kept = [(x, y, radius) for x, y, radius, _color in self.holes]
        self.holes.clear()
        for x, y, radius in kept:
            self.add_hole(x, y, radius)

    def add_hole(self, x: float, y: float, radius: int | None = None) -> tuple[int, int]:
        if radius is None:
            radius = max(6, int(round(self.outer_r * 0.028)))
        radius = int(radius)
        height, width = self.plain.shape[:2]
        x = int(np.clip(round(x), radius + 2, width - radius - 3))
        y = int(np.clip(round(y), radius + 2, height - radius - 3))
        bg = float(self.plain[y, x].mean())
        color = (24, 24, 24) if bg > 110 else (235, 235, 235)
        self.holes.append((x, y, radius, color))
        return x, y

    def add_random_hole(self) -> tuple[int, int]:
        limit = self.outer_r * 0.85
        dx = float(np.clip(self.rng.normal(0, self.outer_r * 0.18), -limit, limit))
        dy = float(np.clip(self.rng.normal(0, self.outer_r * 0.18), -limit, limit))
        return self.add_hole(self.center[0] + dx, self.center[1] + dy)

    def frame(self) -> np.ndarray:
        image = self.plain.copy()
        for x, y, radius, color in self.holes:
            cv2.circle(image, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
            rim = (245, 245, 245) if color[0] < 128 else (12, 12, 12)
            cv2.circle(image, (x, y), radius, rim, 1, lineType=cv2.LINE_AA)
        if self.noise <= 0:
            return image
        noise = self.rng.normal(0, self.noise, image.shape)
        beta = float(self.rng.normal(0, 2.0))
        return np.clip(image.astype(np.float32) + noise + beta, 0, 255).astype(np.uint8)
