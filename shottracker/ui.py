"""Тёмная раскладка окна. Кириллица рисуется через системный шрифт."""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG = (16, 12, 10)
PANEL = (28, 21, 18)
PANEL2 = (46, 31, 26)
GOLD = (75, 184, 232)
RED = (68, 68, 255)
GREEN = (136, 255, 68)
BORDER = (64, 48, 42)
TEXT = (224, 208, 200)
MUTED = (144, 117, 106)
INK = (22, 18, 14)

_FONTS: dict[tuple[int, bool], ImageFont.FreeTypeFont] = {}


def _font_path(bold: bool) -> str:
    names = ("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")
    windir = os.environ.get("WINDIR", r"C:\Windows")
    for name in names:
        path = os.path.join(windir, "Fonts", name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Не найден Segoe UI или Arial")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    key = (size, bold)
    cached = _FONTS.get(key)
    if cached is None:
        cached = ImageFont.truetype(_font_path(bold), size)
        _FONTS[key] = cached
    return cached


def text_metrics(text: str, size: int, bold: bool = False) -> tuple[int, int, int, int]:
    box = font(size, bold).getbbox(text)
    return box[2] - box[0], box[3] - box[1], box[0], box[1]


@dataclass
class Button:
    id: str
    x: int
    y: int
    w: int
    h: int

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h


@dataclass
class ViewGeom:
    ox: int
    oy: int
    nw: int
    nh: int
    scale: float
    iw: int
    ih: int
    space: str


def map_view(px: int, py: int, geom: ViewGeom | None) -> tuple[float, float] | None:
    if geom is None or geom.nw <= 0 or geom.scale <= 0:
        return None
    if not (geom.ox <= px < geom.ox + geom.nw and geom.oy <= py < geom.oy + geom.nh):
        return None
    return (px - geom.ox) / geom.scale, (py - geom.oy) / geom.scale


def _fill_round(img: np.ndarray, x: int, y: int, w: int, h: int, color: tuple[int, int, int], radius: int) -> None:
    if w < 4 or h < 4:
        return
    radius = int(min(radius, w // 2, h // 2))
    cv2.rectangle(img, (x + radius, y), (x + w - radius - 1, y + h - 1), color, -1)
    cv2.rectangle(img, (x, y + radius), (x + w - 1, y + h - radius - 1), color, -1)
    cv2.circle(img, (x + radius, y + radius), radius, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (x + w - radius - 1, y + radius), radius, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (x + radius, y + h - radius - 1), radius, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (x + w - radius - 1, y + h - radius - 1), radius, color, -1, lineType=cv2.LINE_AA)


@dataclass
class _Text:
    x: int
    y: int
    text: str
    size: int
    fill: tuple[int, int, int]
    bold: bool


class Canvas:
    def __init__(self, width: int, height: int):
        self.img = np.full((height, width, 3), BG, np.uint8)
        self.texts: list[_Text] = []
        self.buttons: list[Button] = []

    def fill(self, x: int, y: int, w: int, h: int, color: tuple[int, int, int]) -> None:
        height, width = self.img.shape[:2]
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(width, int(x + w)), min(height, int(y + h))
        if x2 > x1 and y2 > y1:
            self.img[y1:y2, x1:x2] = color

    def rect(self, x: int, y: int, w: int, h: int, color: tuple[int, int, int], thickness: int = 1) -> None:
        cv2.rectangle(
            self.img,
            (int(x), int(y)),
            (int(x + w - 1), int(y + h - 1)),
            color,
            thickness,
        )

    def round_box(
        self, x: int, y: int, w: int, h: int,
        fill: tuple[int, int, int],
        border: tuple[int, int, int],
        radius: int = 8,
    ) -> None:
        _fill_round(self.img, int(x), int(y), int(w), int(h), border, radius)
        _fill_round(self.img, int(x) + 1, int(y) + 1, int(w) - 2, int(h) - 2, fill, max(1, radius - 1))

    def text(self, x: float, y: float, text: str, size: int, bgr: tuple[int, int, int], bold: bool = False) -> None:
        self.texts.append(_Text(int(x), int(y), text, size, (int(bgr[2]), int(bgr[1]), int(bgr[0])), bold))

    def button(
        self, ident: str, label: str, x: int, y: int, w: int, h: int, active: bool = False,
    ) -> Button:
        fill = GOLD if active else PANEL2
        border = GOLD if active else BORDER
        fg = INK if active else TEXT
        self.round_box(x, y, w, h, fill, border, radius=8)
        tw, th, left, top = text_metrics(label, 14, True)
        self.text(x + (w - tw) / 2 - left, y + (h - th) / 2 - top, label, 14, fg, True)
        button = Button(ident, int(x), int(y), int(w), int(h))
        self.buttons.append(button)
        return button

    def finish(self) -> np.ndarray:
        if not self.texts:
            return self.img
        rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        for item in self.texts:
            draw.text((item.x, item.y), item.text, font=font(item.size, item.bold), fill=item.fill)
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def blit_contain(dst: np.ndarray, src: np.ndarray, box: tuple[int, int, int, int]) -> tuple[int, int, int, int, float]:
    x, y, w, h = box
    ih, iw = src.shape[:2]
    if iw <= 0 or ih <= 0 or w <= 0 or h <= 0:
        return x, y, 0, 0, 1.0
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(src, (nw, nh), interpolation=interp)
    ox = x + (w - nw) // 2
    oy = y + (h - nh) // 2
    dst[oy:oy + nh, ox:ox + nw] = resized
    return ox, oy, nw, nh, scale
