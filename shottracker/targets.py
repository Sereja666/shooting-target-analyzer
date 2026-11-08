"""Типы мишеней и подсчёт очков.

Радиусы взяты из HTML-демо и нормированы: внешний край зоны 1 равен 1.
X на десятичной мишени даёт 10 очков, на биатлоне — 5.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import atan2, hypot, pi

import cv2
import numpy as np

GOLD = (75, 184, 232)


@dataclass(frozen=True)
class Ring:
    score: int | str
    radius: float


@dataclass(frozen=True)
class TargetType:
    key: str
    name: str
    kind: str
    rings: tuple[Ring, ...]


def _rings(pairs: list[tuple[int | str, float]]) -> tuple[Ring, ...]:
    outer = pairs[-1][1]
    return tuple(Ring(score, radius / outer) for score, radius in pairs)


TARGETS: dict[str, TargetType] = {
    "pistol": TargetType(
        "pistol",
        "Пистолетная ИСС",
        "issf",
        _rings([
            ("X", 0.05), (10, 0.09), (9, 0.14), (8, 0.21), (7, 0.28),
            (6, 0.37), (5, 0.47), (4, 0.57), (3, 0.68), (2, 0.78), (1, 0.88),
        ]),
    ),
    "rifle": TargetType(
        "rifle",
        "Винтовочная 300м",
        "issf",
        _rings([
            ("X", 0.04), (10, 0.08), (9, 0.14), (8, 0.22), (7, 0.30),
            (6, 0.40), (5, 0.50), (4, 0.62), (3, 0.74), (2, 0.86), (1, 0.96),
        ]),
    ),
    "airgun": TargetType(
        "airgun",
        "Пневматика 10м",
        "issf",
        _rings([
            ("X", 0.04), (10, 0.07), (9, 0.13), (8, 0.20), (7, 0.28),
            (6, 0.38), (5, 0.49), (4, 0.60), (3, 0.73), (2, 0.84), (1, 0.95),
        ]),
    ),
    "biathlon": TargetType(
        "biathlon",
        "Биатлон",
        "biathlon",
        _rings([
            ("X", 0.12), (5, 0.25), (4, 0.40), (3, 0.57), (2, 0.76), (1, 0.93),
        ]),
    ),
    "silhouette": TargetType(
        "silhouette",
        "Силуэт",
        "silhouette",
        _rings([
            (5, 0.15), (4, 0.30), (3, 0.50), (2, 0.70), (1, 0.90),
        ]),
    ),
}

TARGET_ORDER = ("pistol", "rifle", "airgun", "biathlon", "silhouette")


def score_at(norm_x: float, norm_y: float, target: TargetType) -> int | str:
    dist = hypot(norm_x, norm_y)
    for ring in target.rings:
        if dist <= ring.radius + 1e-6:
            return ring.score
    return 0


def score_points(score: int | str, target: TargetType) -> int:
    if score == "X":
        numeric = [ring.score for ring in target.rings if isinstance(ring.score, int)]
        return max(numeric) if numeric else 0
    return int(score)


@dataclass
class Hit:
    norm_x: float
    norm_y: float
    score: int | str
    stamp: datetime
    source: str
    frame_i: int = -1

    def rescore(self, target: TargetType) -> None:
        self.score = score_at(self.norm_x, self.norm_y, target)


def deviation_pct(hit: Hit) -> float:
    """0% — центр мишени, 100% — внешний край зоны 1."""
    return hypot(hit.norm_x, hit.norm_y) * 100.0


def group_report(hits: list[Hit]) -> dict | None:
    if len(hits) < 2:
        return None
    xs = [hit.norm_x for hit in hits]
    ys = [hit.norm_y for hit in hits]
    avg_x = sum(xs) / len(xs)
    avg_y = sum(ys) / len(ys)
    spread = hypot(max(xs) - min(xs), max(ys) - min(ys))
    centered = abs(avg_x) <= 0.05 and abs(avg_y) <= 0.05
    return {
        "count": len(hits),
        "spread": spread,
        "avg_x": avg_x,
        "avg_y": avg_y,
        "centered": centered,
        "direction": "в центре" if centered else _direction(avg_x, avg_y),
    }


def _direction(avg_x: float, avg_y: float) -> str:
    angle = atan2(avg_y, avg_x) * 180.0 / pi
    if -22.5 < angle <= 22.5:
        return "право"
    if 22.5 < angle <= 67.5:
        return "право-низ"
    if 67.5 < angle <= 112.5:
        return "низ"
    if 112.5 < angle <= 157.5:
        return "лево-низ"
    if angle > 157.5 or angle <= -157.5:
        return "лево"
    if -157.5 < angle <= -112.5:
        return "лево-верх"
    if -112.5 < angle <= -67.5:
        return "верх"
    return "право-верх"


def fill_bgr(target: TargetType, ring: Ring) -> tuple[int, int, int]:
    if target.kind == "issf":
        if ring.score == "X" or (isinstance(ring.score, int) and ring.score >= 7):
            return (22, 22, 22)
        return (236, 236, 236)
    if target.kind == "biathlon":
        if ring.score == "X" or (isinstance(ring.score, int) and ring.score >= 4):
            return (32, 32, 32)
        return (230, 230, 230)
    lightness = int(220 - 130 * ring.radius)
    return (int(lightness * 0.55), int(lightness * 0.72), lightness)


def paint_rings(img: np.ndarray, cx: float, cy: float, radius: float, target: TargetType) -> None:
    center = (int(round(cx)), int(round(cy)))
    outer = int(round(radius))
    if outer <= 2:
        return
    rings_out = list(reversed(target.rings))
    for ring in rings_out:
        rr = int(round(ring.radius * radius))
        if rr > 0:
            cv2.circle(img, center, rr, fill_bgr(target, ring), -1, lineType=cv2.LINE_AA)
    for ring in target.rings:
        rr = int(round(ring.radius * radius))
        if rr <= 2:
            continue
        fill = fill_bgr(target, ring)
        line = (230, 230, 230) if sum(fill) < 380 else (25, 25, 25)
        cv2.circle(img, center, rr, line, 1, lineType=cv2.LINE_AA)
    _draw_labels(img, center[0], center[1], radius, target)
    cv2.line(img, (center[0] - 8, center[1]), (center[0] + 8, center[1]), (150, 150, 150), 1, lineType=cv2.LINE_AA)
    cv2.line(img, (center[0], center[1] - 8), (center[0], center[1] + 8), (150, 150, 150), 1, lineType=cv2.LINE_AA)


def _draw_labels(img: np.ndarray, cx: int, cy: int, radius: float, target: TargetType) -> None:
    rings = list(target.rings)
    for index, ring in enumerate(rings):
        if ring.score == "X":
            continue
        rr = ring.radius * radius
        if rr < 22:
            continue
        inner = rings[index - 1].radius * radius if index else 0.0
        label_r = (rr + inner) / 2.0
        text = str(ring.score)
        fill = fill_bgr(target, ring)
        color = (245, 245, 245) if sum(fill) < 380 else (20, 20, 20)
        cv2.putText(
            img, text, (cx - 5, int(cy - label_r) + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA,
        )
    if rings and rings[0].score == "X":
        xr = max(10.0, rings[0].radius * radius)
        fill = fill_bgr(target, rings[0])
        color = (245, 245, 245) if sum(fill) < 380 else (20, 20, 20)
        cv2.putText(
            img, "X", (cx - 6, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX, max(0.4, xr / 48.0), color, 1, cv2.LINE_AA,
        )


def draw_hits(
    img: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    hits: list[Hit],
    highlight: int | None,
    show_poi: bool = False,
) -> None:
    for index, hit in enumerate(hits):
        x = int(round(cx + hit.norm_x * radius))
        y = int(round(cy + hit.norm_y * radius))
        hot = highlight == index
        rr = max(5, min(12, int(radius * 0.035)))
        if hot:
            rr += 2
        cv2.circle(img, (x, y), rr, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (x, y), rr, (40, 220, 255) if hot else (70, 110, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(
            img, str(index + 1), (x + rr, y - rr),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 160, 255), 1, cv2.LINE_AA,
        )
    if show_poi and len(hits) >= 2:
        mx = sum(hit.norm_x for hit in hits) / len(hits)
        my = sum(hit.norm_y for hit in hits) / len(hits)
        cv2.drawMarker(
            img,
            (int(round(cx + mx * radius)), int(round(cy + my * radius))),
            GOLD,
            cv2.MARKER_TILTED_CROSS,
            16,
            2,
            cv2.LINE_AA,
        )


def export_text(
    hits: list[Hit],
    target: TargetType,
    group: dict | None,
    note: str = "",
    videos: list[str] | None = None,
) -> str:
    lines = [
        "ShotTracker — результаты",
        f"Мишень: {target.name}",
        f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}",
        "—" * 52,
        f"{'№':>3}  {'Очки':<6} {'X':>7} {'Y':>7} {'Откл':>6}  Время     Источник",
    ]
    source_label = {"camera": "камера", "manual": "вручную"}
    for index, hit in enumerate(hits, start=1):
        if hit.score == "X":
            label = "X"
        elif hit.score == 0:
            label = "мимо"
        else:
            label = str(hit.score)
        lines.append(
            f"{index:3}  {label:<6} {hit.norm_x:7.3f} {hit.norm_y:7.3f} {deviation_pct(hit):5.0f}%  "
            f"{hit.stamp.strftime('%H:%M:%S')}  {source_label.get(hit.source, hit.source)}"
        )
    total = sum(score_points(hit.score, target) for hit in hits)
    lines.append("—" * 52)
    lines.append(f"Выстрелов: {len(hits)}")
    lines.append(f"Сумма:     {total}")
    if hits:
        lines.append(f"Среднее:   {total / len(hits):.2f}")
        devs = [deviation_pct(hit) for hit in hits]
        lines.append(f"Отклонение: {sum(devs) / len(devs):.0f}% среднее, лучший {min(devs):.0f}%")
    lines.append(f"X-ring:    {sum(1 for hit in hits if hit.score == 'X')}")
    if group:
        lines.append(f"Разброс:   {group['spread']:.3f}")
        lines.append(f"СТП:       {group['avg_x']:+.3f} / {group['avg_y']:+.3f}")
        lines.append(f"Смещение:  {group['direction']}")
    if note.strip():
        lines.append("")
        lines.append("Заметка:")
        lines.append(note.strip())
    if videos:
        lines.append("")
        lines.append("Видео:")
        lines.extend(videos)
    lines.append("")
    return "\n".join(lines)
