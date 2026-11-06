"""Новое отверстие — маленькое круглое пятно, которое остаётся на месте.

Вспышка, человек в кадре и дрожание бумаги отсекаются: пятно должно
продержаться несколько кадров, а сильный сдвиг кадра пропускается.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot, pi

import cv2
import numpy as np

from shottracker.geometry import translate

ACC_FRAMES = 4
MAX_SHIFT = 24.0
MIN_RESPONSE = 0.05

_WINDOWS: dict[tuple[int, int], np.ndarray] = {}


def _window(height: int, width: int) -> np.ndarray:
    key = (height, width)
    window = _WINDOWS.get(key)
    if window is None:
        window = cv2.createHanningWindow((width, height), cv2.CV_32F)
        _WINDOWS[key] = window
    return window


def alignment_delta(reference: np.ndarray, current: np.ndarray) -> tuple[float, float, float]:
    """Сдвиг, который через translate() кладёт current на reference."""
    height, width = reference.shape
    shift, response = cv2.phaseCorrelate(
        reference.astype(np.float32),
        current.astype(np.float32),
        _window(height, width),
    )
    return -float(shift[0]), -float(shift[1]), float(response)


def blob_limits(outer_r: float, scale: float) -> tuple[float, float]:
    scale = float(np.clip(scale, 0.4, 2.5))
    min_r = max(2.0, outer_r * 0.009 * scale)
    max_r = max(min_r + 4.0, outer_r * 0.15 * scale)
    return pi * min_r * min_r, pi * max_r * max_r


@dataclass
class Blob:
    x: float
    y: float
    r: float
    stable: int = 1


@dataclass
class Candidate:
    x: float
    y: float
    r: float
    stable: int
    misses: int


@dataclass
class ProcessResult:
    status: str
    new_holes: list[Blob] = field(default_factory=list)
    watching: list[Blob] = field(default_factory=list)
    mask: np.ndarray | None = None
    align_dx: float = 0.0
    align_dy: float = 0.0


class HoleDetector:
    def __init__(self) -> None:
        self.sensitivity = 28
        self.min_area = 20.0
        self.max_area = 2500.0
        self.min_circularity = 0.42
        self.persistence = 4
        self.match_radius = 18.0
        self.reference: np.ndarray | None = None
        self.candidates: list[Candidate] = []
        self.confirmed: list[tuple[float, float, float]] = []
        self._acc: np.ndarray | None = None
        self._acc_n = 0

    def reset(self) -> None:
        self.reference = None
        self.candidates.clear()
        self.confirmed.clear()
        self._acc = None
        self._acc_n = 0

    def process(self, bgr: np.ndarray) -> ProcessResult:
        gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        if self.reference is not None and self.reference.shape != gray.shape:
            self.reset()

        if self.reference is None:
            if self._acc is None or self._acc.shape != gray.shape:
                self._acc = gray.astype(np.float32)
                self._acc_n = 1
            else:
                self._acc += gray
                self._acc_n += 1
            if self._acc_n < ACC_FRAMES:
                return ProcessResult("warmup")
            self.reference = np.clip(self._acc / self._acc_n, 0, 255).astype(np.uint8)
            self._acc = None
            self._acc_n = 0
            return ProcessResult("ready")

        dx, dy, response = alignment_delta(self.reference, gray)
        if abs(dx) > MAX_SHIFT or abs(dy) > MAX_SHIFT or (response < MIN_RESPONSE and hypot(dx, dy) > 8):
            self._age()
            return ProcessResult("skip-shift", watching=self._watching())
        if response < MIN_RESPONSE:
            dx, dy = 0.0, 0.0
        aligned = translate(gray, dx, dy) if (dx or dy) else gray

        ref = self.reference.astype(np.float32)
        cur = aligned.astype(np.float32)
        delta = float(np.median(cur[::4, ::4] - ref[::4, ::4]))
        change = np.abs(ref - (cur - delta))
        change_u8 = np.clip(change, 0, 255).astype(np.uint8)
        _, mask = cv2.threshold(change_u8, int(self.sensitivity), 255, cv2.THRESH_BINARY)
        mask = _clean_mask(mask)

        fraction = float(np.count_nonzero(mask)) / float(mask.size)
        if fraction > 0.035:
            self._age()
            return ProcessResult("skip-motion", watching=self._watching(), mask=mask, align_dx=dx, align_dy=dy)

        blobs = _blobs(mask, self.min_area, self.max_area, self.min_circularity)
        if len(blobs) > 10:
            self._age()
            return ProcessResult("skip-noise", watching=self._watching(), mask=mask, align_dx=dx, align_dy=dy)

        fresh = self._update_tracks(blobs, aligned)
        return ProcessResult("ok", fresh, self._watching(), mask, dx, dy)

    def _watching(self) -> list[Blob]:
        return [Blob(c.x, c.y, c.r, c.stable) for c in self.candidates if c.stable >= 2]

    def _age(self) -> None:
        for cand in self.candidates:
            cand.misses += 1
        self.candidates = [cand for cand in self.candidates if cand.misses <= 2]

    def _near_confirmed(self, x: float, y: float, radius: float) -> bool:
        limit = max(12.0, radius * 2.5)
        return any(hypot(x - cx, y - cy) < max(limit, cr * 2.5) for cx, cy, cr in self.confirmed)

    def _absorb(self, aligned_gray: np.ndarray, x: float, y: float, radius: float) -> None:
        if self.reference is None:
            return
        paint = np.zeros(self.reference.shape, np.uint8)
        cv2.circle(paint, (int(round(x)), int(round(y))), int(round(radius)) + 7, 255, -1)
        updated = self.reference.copy()
        updated[paint > 0] = aligned_gray[paint > 0]
        self.reference = updated

    def _update_tracks(self, blobs: list[Blob], aligned_gray: np.ndarray) -> list[Blob]:
        blobs = [blob for blob in blobs if not self._near_confirmed(blob.x, blob.y, blob.r)]
        pairs: list[tuple[float, int, int]] = []
        for i, blob in enumerate(blobs):
            for j, cand in enumerate(self.candidates):
                dist = hypot(blob.x - cand.x, blob.y - cand.y)
                if dist <= self.match_radius:
                    pairs.append((dist, i, j))
        pairs.sort()
        used_b: set[int] = set()
        used_c: set[int] = set()
        for _, i, j in pairs:
            if i in used_b or j in used_c:
                continue
            used_b.add(i)
            used_c.add(j)
            cand = self.candidates[j]
            cand.x = cand.x * 0.35 + blobs[i].x * 0.65
            cand.y = cand.y * 0.35 + blobs[i].y * 0.65
            cand.r = blobs[i].r
            cand.stable += 1
            cand.misses = 0
        for j, cand in enumerate(self.candidates):
            if j not in used_c:
                cand.misses += 1
        for i, blob in enumerate(blobs):
            if i not in used_b:
                self.candidates.append(Candidate(blob.x, blob.y, blob.r, stable=1, misses=0))

        fresh: list[Blob] = []
        keep: list[Candidate] = []
        for cand in self.candidates:
            if cand.misses > 2:
                continue
            if cand.stable >= self.persistence and cand.misses == 0:
                if not self._near_confirmed(cand.x, cand.y, cand.r):
                    fresh.append(Blob(cand.x, cand.y, cand.r, cand.stable))
                    self.confirmed.append((cand.x, cand.y, cand.r))
                    self._absorb(aligned_gray, cand.x, cand.y, cand.r)
                continue
            keep.append(cand)
        if len(keep) > 40:
            keep = keep[-40:]
        self.candidates = keep
        return fresh


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask[:10, :] = 0
    mask[-10:, :] = 0
    mask[:, :10] = 0
    mask[:, -10:] = 0
    return mask


def _blobs(mask: np.ndarray, min_area: float, max_area: float, min_circ: float) -> list[Blob]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs: list[Blob] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 1e-3:
            continue
        circularity = 4.0 * pi * area / (perimeter * perimeter)
        if circularity < min_circ:
            continue
        moment = cv2.moments(contour)
        if moment["m00"] <= 1e-3:
            continue
        blobs.append(Blob(
            float(moment["m10"] / moment["m00"]),
            float(moment["m01"] / moment["m00"]),
            float((area / pi) ** 0.5),
        ))
    blobs.sort(key=lambda blob: blob.r, reverse=True)
    return blobs[:6]
