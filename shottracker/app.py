"""Окно ShotTracker: камера, калибровка, попадания, очки и группа."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from datetime import datetime
from math import hypot
from pathlib import Path

import cv2
import numpy as np

from shottracker.demo import DemoWorld
from shottracker.detector import HoleDetector, blob_limits
from shottracker.geometry import (
    Calibration,
    find_circle,
    load_calibration,
    quad_ok,
    save_calibration,
    translate,
    warp_frame,
    work_view,
    WorkView,
)
from shottracker.targets import (
    TARGET_ORDER,
    TARGETS,
    Hit,
    draw_hits,
    export_text,
    group_report,
    paint_rings,
    score_at,
    score_points,
)
from shottracker.ui import (
    BG,
    BORDER,
    GOLD,
    GREEN,
    INK,
    MUTED,
    PANEL,
    PANEL2,
    RED,
    TEXT,
    Canvas,
    ViewGeom,
    blit_contain,
    map_view,
    text_metrics,
)

# geometry.ViewGeom does not exist — ViewGeom lives in ui. Remove the bad import.
# The duplicate import above is a mistake I must not leave. I'll fix after writing
# by not including it. Checking the file... I included a bad import. I'll rewrite
# the import block cleanly in a follow-up if this file is saved with it.
# Actually I'll just write the correct file. Let me not include the bad import.

ROOT = Path(__file__).resolve().parent.parent
CALIB_PATH = ROOT / "calibration.json"
RESULTS = ROOT / "results"
WIN = "ShotTracker"
WIN_W, WIN_H = 1280, 780
HEADER_H, FOOTER_H = 56, 54

CORNER_NAMES = ("левый верх листа", "правый верх листа", "правый низ листа", "левый низ листа")


def _console_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def play_tone(score: int | str) -> None:
    if score == "X" or score == 10:
        freq = 1170
    elif isinstance(score, int) and score >= 9:
        freq = 880
    elif isinstance(score, int) and score >= 7:
        freq = 660
    elif score == 0:
        freq = 220
    else:
        freq = 440

    def _run() -> None:
        try:
            import winsound
            winsound.Beep(freq, 80)
        except (RuntimeError, ValueError, ImportError):
            pass

    threading.Thread(target=_run, daemon=True).start()


def downscale(frame: np.ndarray, max_w: int = 1280) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= max_w:
        return frame
    scale = max_w / width
    return cv2.resize(frame, (max_w, int(height * scale)), interpolation=cv2.INTER_AREA)


class ShotApp:
    def __init__(self, world: DemoWorld | None = None, sound: bool = True, source_name: str = "Камера"):
        self.world = world
        self.sound = sound
        self.source_name = "Демо" if world is not None else source_name
        self.detector = HoleDetector()
        self.target_key = "pistol"
        self.hits: list[Hit] = []
        self._cleared: list[Hit] = []
        self.analysis = False
        self.mode = "live"
        self.pending_corners: list[tuple[float, float]] = []
        self.pending_space = "raw"
        self.pending_center: tuple[float, float] | None = None
        self.calib: Calibration | None = None
        self.size_scale = 1.0
        self.show_mask = False
        self.buttons: list = []
        self.view_geom: ViewGeom | None = None
        self.schem_geom: tuple[float, float, float, tuple[int, int, int, int]] | None = None
        self.hit_rows: list[tuple[tuple[int, int, int, int], int]] = []
        self.mouse: tuple[int, int] | None = None
        self.raw_frame: np.ndarray | None = None
        self.frame_size: tuple[int, int] | None = None
        self.last_result = None
        self.last_view = None
        self._announce = ""
        self._announce_until = 0.0
        self.flash_until = 0.0
        self.highlight: int | None = None
        self.highlight_until = 0.0
        self._next_shot: float | None = None
        if world is not None:
            self.target_key = world.target_key
            self.calib = Calibration("raw", world.center, world.outer_r, None, (world.w, world.h))
            self.analysis = True
        else:
            loaded = load_calibration(CALIB_PATH)
            if loaded:
                self.calib, key = loaded
                if key in TARGETS:
                    self.target_key = key

    def announce(self, text: str, ttl: float = 3.0) -> None:
        self._announce = text
        self._announce_until = time.time() + ttl

    def overlay_message(self) -> str:
        now = time.time()
        if self._announce and now < self._announce_until:
            return self._announce
        if self.mode == "corners":
            return "Правый клик — отмена"
        if self.mode == "center":
            return "Кликните центр. Правый клик — отмена"
        if self.mode == "edge":
            return "Кликните край зоны 1. Правый клик — отмена"
        if self.calib is None:
            return "Нажмите «Центр» и отметьте мишень"
        if not self.analysis:
            return "Анализ на паузе"
        result = self.last_result
        if result is None:
            return "Анализ включён"
        if result.status == "ok" and result.watching:
            return "Вижу новое пятно, подтверждаю…"
        return {
            "warmup": "Снимаю эталон, не стреляйте секунду…",
            "ready": "Эталон снят. Можно стрелять.",
            "skip-shift": "Сильный сдвиг камеры — кадр пропущен",
            "skip-motion": "В кадре движение — кадр пропущен",
            "skip-noise": "Много пятен. Поднимите порог",
            "ok": "Анализ идёт",
        }.get(result.status, "Анализ идёт")

    def calib_banner(self) -> str:
        if self.mode == "corners":
            index = min(len(self.pending_corners), 3)
            return f"Угол {index + 1}/4: {CORNER_NAMES[index]}"
        if self.mode == "center":
            return "Кликните центр мишени"
        if self.mode == "edge":
            return "Кликните внешний край зоны 1"
        return ""

    def start_center(self) -> None:
        self.mode = "center"
        self.pending_space = "raw"
        self.pending_corners = []
        self.pending_center = None
        self.announce("Кликните центр мишени", 8)

    def start_corners(self) -> None:
        self.mode = "corners"
        self.pending_space = "raw"
        self.pending_corners = []
        self.pending_center = None
        self.announce("Четыре угла листа по часовой, с левого верхнего", 8)

    def cancel_calib(self) -> None:
        if self.mode == "live":
            return
        self.mode = "live"
        self.pending_corners = []
        self.pending_center = None
        self.announce("Калибровка отменена")

    def take_reference(self) -> None:
        self.detector.reset()
        self._next_shot = None
        if self.analysis and self.calib is not None and self.mode == "live":
            self.announce("Снимаю новый эталон…")
        else:
            self.announce("Эталон обновится, когда идёт анализ")

    def toggle_analysis(self) -> None:
        if self.calib is None:
            self.announce("Сначала откалибруйте мишень")
            return
        self.analysis = not self.analysis
        self.announce("Анализ включён" if self.analysis else "Анализ на паузе")

    def set_target(self, key: str) -> None:
        if key not in TARGETS:
            return
        self.target_key = key
        target = TARGETS[key]
        for hit in self.hits:
            hit.rescore(target)
        if self.world is not None:
            self.world.set_target(key)
            self.detector.reset()
            self._next_shot = None
        self._save_calib()
        self.announce(target.name)

    def cycle_target(self) -> None:
        index = TARGET_ORDER.index(self.target_key) if self.target_key in TARGET_ORDER else 0
        self.set_target(TARGET_ORDER[(index + 1) % len(TARGET_ORDER)])

    def demo_shot(self) -> None:
        if self.world is None:
            self.announce("Вживую выстрел появится сам. Пробел — только в демо")
            return
        if self.detector.reference is None:
            self.announce("Подождите, пока снимется эталон")
            return
        self.world.add_random_hole()

    def undo(self) -> None:
        if self._cleared:
            self.hits = self._cleared
            self._cleared = []
            self.announce("Сброс отменён")
            return
        if not self.hits:
            self.announce("Нечего отменять")
            return
        self.hits.pop()
        self.announce("Последний выстрел убран")

    def clear_hits(self) -> None:
        if not self.hits:
            self.announce("Список уже пуст")
            return
        self._cleared = list(self.hits)
        self.hits = []
        self.announce("Попадания очищены. «Отмена» вернёт их")

    def export(self) -> None:
        if not self.hits:
            self.announce("Нет данных для экспорта")
            return
        RESULTS.mkdir(exist_ok=True)
        path = RESULTS / f"shottracker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        target = TARGETS[self.target_key]
        path.write_text(export_text(self.hits, target, group_report(self.hits)), encoding="utf-8")
        print(f"Сохранено: {path}")
        self.announce(f"Сохранено {path.name}")

    def auto_circle(self) -> None:
        if self.raw_frame is None:
            return
        frame = self.raw_frame
        corners = None
        space = "raw"
        image = frame
        frame_size = (frame.shape[1], frame.shape[0])
        if self.calib is not None and self.calib.space == "warp" and self.calib.corners:
            calib = self.calib.scaled_to(frame_size[0], frame_size[1])
            image = warp_frame(frame, calib.corners)
            corners = calib.corners
            space = "warp"
            frame_size = calib.frame_size or frame_size
        found = find_circle(image)
        if found is None:
            self.announce("Круг не найден — отметьте центр и край вручную")
            return
        cx, cy, radius = found
        self.calib = Calibration(space, (cx, cy), radius, corners, frame_size)
        self.detector.reset()
        self._next_shot = None
        self.mode = "live"
        self._save_calib()
        self.announce("Круг найден. Проверьте жёлтое кольцо")

    def on_button(self, ident: str) -> None:
        if ident == "center":
            self.start_center()
        elif ident == "corners":
            self.start_corners()
        elif ident == "auto":
            self.auto_circle()
        elif ident == "reference":
            self.take_reference()
        elif ident == "analysis":
            self.toggle_analysis()
        elif ident == "export":
            self.export()
        elif ident == "clear":
            self.clear_hits()
        elif ident == "undo":
            self.undo()
        elif ident == "shot":
            self.demo_shot()
        elif ident == "target":
            self.cycle_target()
        elif ident == "sens-":
            self.detector.sensitivity = max(8, self.detector.sensitivity - 2)
        elif ident == "sens+":
            self.detector.sensitivity = min(80, self.detector.sensitivity + 2)
        elif ident == "size-":
            self.size_scale = max(0.4, round(self.size_scale - 0.1, 2))
        elif ident == "size+":
            self.size_scale = min(2.5, round(self.size_scale + 0.1, 2))
        elif ident == "mask":
            self.show_mask = not self.show_mask

    def on_key(self, key: int) -> bool:
        if key in (27, ord("q"), ord("Q")):
            return False
        actions = {
            ord("c"): self.start_center,
            ord("C"): self.start_center,
            ord("p"): self.start_corners,
            ord("P"): self.start_corners,
            ord("f"): self.auto_circle,
            ord("F"): self.auto_circle,
            ord("r"): self.take_reference,
            ord("R"): self.take_reference,
            ord("a"): self.toggle_analysis,
            ord("A"): self.toggle_analysis,
            ord("e"): self.export,
            ord("E"): self.export,
            ord("x"): self.clear_hits,
            ord("X"): self.clear_hits,
            ord("z"): self.undo,
            ord("Z"): self.undo,
            ord(" "): self.demo_shot,
            ord("m"): self._toggle_mask,
            ord("M"): self._toggle_mask,
            ord("-"): lambda: self.on_button("sens-"),
            ord("["): lambda: self.on_button("sens-"),
            ord("="): lambda: self.on_button("sens+"),
            ord("]"): lambda: self.on_button("sens+"),
        }
        if key in actions:
            actions[key]()
        elif ord("1") <= key <= ord("5"):
            self.set_target(TARGET_ORDER[key - ord("1")])
        return True

    def _toggle_mask(self) -> None:
        self.show_mask = not self.show_mask

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _param) -> None:
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse = (x, y)
            return
        if event == cv2.EVENT_RBUTTONDOWN:
            self.cancel_calib()
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for button in self.buttons:
            if button.contains(x, y):
                self.on_button(button.id)
                return
        if self.mode != "live":
            mapped = map_view(x, y, self.view_geom)
            if mapped is not None:
                self._calib_click(*mapped)
            return
        for rect, index in self.hit_rows:
            rx, ry, rw, rh = rect
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                self.highlight = index
                self.highlight_until = time.time() + 1.6
                return
        self._manual_click(x, y)

    def _calib_click(self, x: float, y: float) -> None:
        if self.mode == "corners":
            self.pending_corners.append((x, y))
            if len(self.pending_corners) < 4:
                self.announce(f"Угол {len(self.pending_corners) + 1}/4: {CORNER_NAMES[len(self.pending_corners)]}", 8)
                return
            if not quad_ok(self.pending_corners):
                self.pending_corners = []
                self.announce("Четырёхугольник кривой, начните снова")
                return
            self.pending_space = "warp"
            self.mode = "center"
            self.announce("Лист выровнен. Теперь центр мишени", 8)
            return
        if self.mode == "center":
            self.pending_center = (x, y)
            self.mode = "edge"
            self.announce("Теперь внешний край зоны 1", 8)
            return
        if self.mode != "edge" or self.pending_center is None or self.frame_size is None:
            return
        cx, cy = self.pending_center
        radius = hypot(x - cx, y - cy)
        if radius < 20:
            self.announce("Край слишком близко к центру")
            return
        corners = list(self.pending_corners) if self.pending_space == "warp" else None
        self.calib = Calibration(self.pending_space, (cx, cy), radius, corners, self.frame_size)
        self.detector.reset()
        self._next_shot = None
        self.mode = "live"
        self.pending_corners = []
        self.pending_center = None
        self._save_calib()
        self.announce("Калибровка сохранена. Включите анализ перед серией")

    def _manual_click(self, x: int, y: int) -> None:
        if self.schem_geom is not None:
            cx, cy, radius, bounds = self.schem_geom
            bx, by, bw, bh = bounds
            if bx <= x <= bx + bw and by <= y <= by + bh:
                norm_x = (x - cx) / radius
                norm_y = (y - cy) / radius
                if hypot(norm_x, norm_y) > 1.15:
                    self.announce("Клик вне мишени")
                    return
                self._add_hit(norm_x, norm_y, "manual")
                return
        if self.view_geom is None or self.view_geom.space != "work" or self.last_view is None:
            return
        mapped = map_view(x, y, self.view_geom)
        if mapped is None or self.last_view.outer_r <= 1:
            return
        norm_x = (mapped[0] - self.last_view.cx) / self.last_view.outer_r
        norm_y = (mapped[1] - self.last_view.cy) / self.last_view.outer_r
        if hypot(norm_x, norm_y) > 1.12:
            return
        self._add_hit(norm_x, norm_y, "manual")

    def _add_hit(self, norm_x: float, norm_y: float, source: str) -> None:
        if any(hypot(hit.norm_x - norm_x, hit.norm_y - norm_y) < 0.012 for hit in self.hits):
            return
        self._cleared = []
        target = TARGETS[self.target_key]
        score = score_at(norm_x, norm_y, target)
        self.hits.append(Hit(norm_x, norm_y, score, datetime.now(), source))
        label = "X" if score == "X" else ("мимо" if score == 0 else str(score))
        self.announce(f"Выстрел {len(self.hits)} · {label}", 2.4)
        self.flash_until = time.time() + 0.45
        self.highlight = len(self.hits) - 1
        self.highlight_until = time.time() + 1.3
        if self.sound:
            play_tone(score)

    def _consume(self, view, result) -> None:
        if result is None or view is None or view.outer_r <= 1:
            return
        for blob in result.new_holes:
            norm_x = (blob.x - view.cx) / view.outer_r
            norm_y = (blob.y - view.cy) / view.outer_r
            if hypot(norm_x, norm_y) > 1.12:
                continue
            self._add_hit(norm_x, norm_y, "camera")

    def _save_calib(self) -> None:
        if self.world is not None or self.calib is None:
            return
        save_calibration(CALIB_PATH, self.calib, self.target_key)

    def step_frame(self, frame: np.ndarray) -> np.ndarray:
        self.raw_frame = frame
        self.frame_size = (frame.shape[1], frame.shape[0])
        view = None
        result = None
        space = "raw"
        show_warp = (
            self.mode in ("center", "edge")
            and self.pending_space == "warp"
            and len(self.pending_corners) == 4
        )
        if show_warp:
            preview = warp_frame(frame, self.pending_corners)
            space = "warp"
            self._draw_pending(preview, space)
        elif self.mode == "live" and self.calib is not None:
            view = work_view(frame, self.calib)
            self.last_view = view
            if self.analysis:
                min_area, max_area = blob_limits(view.outer_r, self.size_scale)
                self.detector.min_area = min_area
                self.detector.max_area = max_area
                result = self.detector.process(view.image)
                self.last_result = result
                if result.status == "ok" and (result.align_dx or result.align_dy):
                    view = WorkView(
                        translate(view.image, result.align_dx, result.align_dy),
                        view.cx, view.cy, view.outer_r,
                    )
                    self.last_view = view
                if result.status == "ready":
                    self.announce("Эталон снят. Можно стрелять.", 2.5)
                self._consume(view, result)
            else:
                self.last_result = None
            self._annotate_work(view, result)
            preview = view.image
            space = "work"
        else:
            preview = frame.copy()
            space = "raw"
            self._draw_pending(preview, space)
            if not self.analysis:
                self.last_result = None
        return self._render(preview, space, view, result)

    def _draw_pending(self, image: np.ndarray, space: str) -> None:
        if self.mode == "corners" and space == "raw":
            pts = [(int(p[0]), int(p[1])) for p in self.pending_corners]
            for point in pts:
                cv2.circle(image, point, 7, GOLD, -1, lineType=cv2.LINE_AA)
            if len(pts) >= 2:
                cv2.polylines(image, [np.array(pts, np.int32)], False, GOLD, 2, lineType=cv2.LINE_AA)
        if self.mode == "edge" and self.pending_center and space == self.pending_space:
            cx, cy = self.pending_center
            cv2.drawMarker(image, (int(cx), int(cy)), GOLD, cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)

    def _annotate_work(self, view, result) -> None:
        image = view.image
        cv2.circle(image, (int(round(view.cx)), int(round(view.cy))), int(round(view.outer_r)), GOLD, 2, lineType=cv2.LINE_AA)
        cv2.drawMarker(image, (int(round(view.cx)), int(round(view.cy))), GOLD, cv2.MARKER_CROSS, 18, 1, cv2.LINE_AA)
        highlight = self.highlight if time.time() < self.highlight_until else None
        draw_hits(image, view.cx, view.cy, view.outer_r, self.hits, highlight, show_poi=False)
        if result is not None:
            for blob in result.watching:
                cv2.circle(
                    image, (int(round(blob.x)), int(round(blob.y))),
                    int(max(8, blob.r + 4)), (0, 220, 255), 2, lineType=cv2.LINE_AA,
                )

    def _render(self, preview: np.ndarray, space: str, view, result) -> np.ndarray:
        canvas = Canvas(WIN_W, WIN_H)
        self._header(canvas)
        cam = (12, HEADER_H + 8, 742, WIN_H - HEADER_H - FOOTER_H - 16)
        self._camera(canvas, cam, preview, space, result)
        panel = (cam[0] + cam[2] + 10, cam[1], WIN_W - (cam[0] + cam[2] + 20), cam[3])
        self._side(canvas, panel, view)
        self._footer(canvas)
        self.buttons = canvas.buttons
        return canvas.finish()

    def _header(self, canvas: Canvas) -> None:
        canvas.fill(0, 0, WIN_W, HEADER_H, PANEL)
        cv2.circle(canvas.img, (28, 28), 12, GOLD, 2, lineType=cv2.LINE_AA)
        cv2.circle(canvas.img, (28, 28), 4, GOLD, -1, lineType=cv2.LINE_AA)
        canvas.text(48, 16, "SHOT", 20, TEXT, True)
        shot_w, _, _, _ = text_metrics("SHOT", 20, True)
        canvas.text(48 + shot_w + 2, 16, "TRACKER", 20, GOLD, True)
        target = TARGETS[self.target_key]
        n = len(self.hits)
        total = sum(score_points(hit.score, target) for hit in self.hits)
        avg = f"{total / n:.1f}" if n else "—"
        best = _best_label(self.hits, target)
        stats = (("Выстрелы", str(n)), ("Сумма", str(total)), ("Среднее", avg), ("Лучший", best))
        x = 330
        for label, value in stats:
            canvas.text(x, 6, value, 20, GOLD, True)
            canvas.text(x, 32, label.upper(), 11, MUTED, False)
            x += 108
        pill = self._pill_text()
        pw, _, _, _ = text_metrics(pill, 13, True)
        pill_w = pw + 36
        px = WIN_W - pill_w - 16
        canvas.round_box(px, 14, pill_w, 28, PANEL2, BORDER, 12)
        dot = GREEN if self.analysis and self.mode == "live" else GOLD if self.mode != "live" else MUTED
        cv2.circle(canvas.img, (px + 14, 28), 5, dot, -1, lineType=cv2.LINE_AA)
        canvas.text(px + 24, 19, pill, 13, TEXT, True)

    def _pill_text(self) -> str:
        if self.mode != "live":
            return f"{self.source_name} · калибровка"
        if self.analysis:
            return f"{self.source_name} · анализ"
        return f"{self.source_name} · пауза"

    def _camera(self, canvas: Canvas, box: tuple[int, int, int, int], preview: np.ndarray, space: str, result) -> None:
        canvas.fill(*box, (0, 0, 0))
        ox, oy, nw, nh, scale = blit_contain(canvas.img, preview, box)
        self.view_geom = ViewGeom(ox, oy, nw, nh, scale, preview.shape[1], preview.shape[0], space)
        if self.mode != "live" and nw > 0:
            canvas.fill(ox, oy, nw, 34, GOLD)
            canvas.text(ox + 12, oy + 8, self.calib_banner(), 15, INK, True)
            self._rubber_band(canvas, ox, oy, scale)
        elif self.analysis and oy > 0:
            cv2.circle(canvas.img, (ox + 16, oy + 16), 5, RED, -1, lineType=cv2.LINE_AA)
            canvas.text(ox + 26, oy + 8, "LIVE", 12, RED, True)
        if self.show_mask and result is not None and result.mask is not None and nw > 180:
            thumb = cv2.cvtColor(result.mask, cv2.COLOR_GRAY2BGR)
            thumb = cv2.resize(thumb, (150, 150), interpolation=cv2.INTER_NEAREST)
            tx, ty = ox + nw - 158, oy + nh - 158
            canvas.img[ty:ty + 150, tx:tx + 150] = thumb
            canvas.rect(tx, ty, 150, 150, GOLD, 1)
        flashing = time.time() < self.flash_until
        canvas.rect(*box, GOLD if flashing else BORDER, 3 if flashing else 1)

    def _rubber_band(self, canvas: Canvas, ox: int, oy: int, scale: float) -> None:
        if self.mode != "edge" or self.pending_center is None or self.mouse is None or self.view_geom is None:
            return
        if self.view_geom.space != self.pending_space:
            return
        sx = ox + self.pending_center[0] * scale
        sy = oy + self.pending_center[1] * scale
        radius = hypot(self.mouse[0] - sx, self.mouse[1] - sy)
        if radius < 2:
            return
        cv2.circle(canvas.img, (int(sx), int(sy)), int(radius), GOLD, 1, lineType=cv2.LINE_AA)

    def _side(self, canvas: Canvas, panel: tuple[int, int, int, int], view) -> None:
        x, y, w, h = panel
        canvas.round_box(x, y, w, h, PANEL, BORDER, 10)
        cursor = y + 10
        canvas.button("target", TARGETS[self.target_key].name, x + 12, cursor, w - 24, 30, False)
        cursor += 36
        canvas.text(x + 14, cursor, self.overlay_message(), 13, GOLD, False)
        cursor += 22
        group_h = 88
        slider_h = 64
        rest = h - (cursor - y) - group_h - slider_h - 28
        schem_h = int(max(200, min(rest * 0.58, 280)))
        schem = (x + 12, cursor, w - 24, schem_h)
        self._schematic(canvas, schem)
        cursor += schem_h + 8
        self._group_box(canvas, x + 12, cursor, w - 24, group_h)
        cursor += group_h + 8
        self._log(canvas, x + 12, cursor, w - 24, max(48, y + h - slider_h - 8 - cursor))
        self._sliders(canvas, x + 12, y + h - slider_h, w - 24)

    def _schematic(self, canvas: Canvas, box: tuple[int, int, int, int]) -> None:
        x, y, w, h = box
        canvas.fill(x, y, w, h, BG)
        cx = x + w / 2.0
        cy = y + h / 2.0
        radius = min(w, h) * 0.46
        self.schem_geom = (cx, cy, radius, box)
        highlight = self.highlight if time.time() < self.highlight_until else None
        paint_rings(canvas.img, cx, cy, radius, TARGETS[self.target_key])
        draw_hits(canvas.img, cx, cy, radius, self.hits, highlight, show_poi=True)

    def _group_box(self, canvas: Canvas, x: int, y: int, w: int, h: int) -> None:
        canvas.round_box(x, y, w, h, PANEL2, BORDER, 8)
        canvas.text(x + 12, y + 6, "ГРУППА", 11, MUTED, True)
        report = group_report(self.hits)
        if report is None:
            canvas.text(x + 12, y + 34, "Нужно минимум два попадания", 14, MUTED, False)
            return
        canvas.text(x + 12, y + 24, f"Разброс   {report['spread']:.3f}", 15, TEXT, False)
        canvas.text(
            x + 12, y + 44,
            f"СТП   {report['avg_x']:+.2f} / {report['avg_y']:+.2f}",
            15, TEXT, False,
        )
        color = GREEN if report["centered"] else RED
        canvas.text(x + 12, y + 64, f"Смещение   {report['direction']}", 15, color, True)
        xcount = sum(1 for hit in self.hits if hit.score == "X")
        canvas.text(x + w - 78, y + 6, f"X {xcount}", 12, GOLD, True)

    def _log(self, canvas: Canvas, x: int, y: int, w: int, h: int) -> None:
        self.hit_rows = []
        if h < 24:
            return
        canvas.text(x, y, "ПОПАДАНИЯ", 11, MUTED, True)
        row_h = 22
        slots = max(0, (h - 18) // row_h)
        if slots <= 0 or not self.hits:
            if not self.hits:
                canvas.text(x, y + 22, "Пока пусто — клик по мишени тоже считается", 13, MUTED, False)
            return
        visible = list(enumerate(self.hits))[-slots:]
        highlight = self.highlight if time.time() < self.highlight_until else None
        for row, (index, hit) in enumerate(reversed(visible)):
            yy = y + 18 + row * row_h
            if highlight == index:
                canvas.fill(x, yy - 1, w, row_h, PANEL2)
            self.hit_rows.append(((x, yy - 1, w, row_h), index))
            canvas.text(x, yy, f"{index + 1:02d}", 13, MUTED, True)
            canvas.text(x + 36, yy, _score_label(hit.score), 14, _score_color(hit.score), True)
            source = "камера" if hit.source == "camera" else "вручную"
            canvas.text(x + 92, yy, f"{hit.norm_x:+.2f} {hit.norm_y:+.2f}   {hit.stamp.strftime('%H:%M:%S')}   {source}", 13, TEXT, False)

    def _sliders(self, canvas: Canvas, x: int, y: int, w: int) -> None:
        canvas.text(x, y + 4, f"Порог {self.detector.sensitivity}", 13, TEXT, True)
        canvas.button("sens-", "−", x + 108, y, 28, 26, False)
        canvas.button("sens+", "+", x + 140, y, 28, 26, False)
        canvas.text(x, y + 36, f"Пятно ×{self.size_scale:.1f}", 13, TEXT, True)
        canvas.button("size-", "−", x + 108, y + 32, 28, 26, False)
        canvas.button("size+", "+", x + 140, y + 32, 28, 26, False)
        canvas.button("mask", "Маска", x + w - 86, y + 32, 86, 26, self.show_mask)

    def _footer(self, canvas: Canvas) -> None:
        canvas.fill(0, WIN_H - FOOTER_H, WIN_W, FOOTER_H, PANEL)
        items = [
            ("center", "Центр", self.mode in ("center", "edge") and self.pending_space == "raw"),
            ("corners", "4 угла", self.mode == "corners" or self.pending_space == "warp" and self.mode != "live"),
            ("auto", "Авто", False),
            ("reference", "Эталон", False),
            ("analysis", "Анализ", self.analysis),
            ("export", "Экспорт", False),
            ("clear", "Сброс", False),
            ("undo", "Отмена", False),
        ]
        if self.world is not None:
            items.append(("shot", "Выстрел", False))
        gap = 8
        widths = []
        for _ident, label, _active in items:
            tw, _, _, _ = text_metrics(label, 14, True)
            widths.append(tw + 28)
        total = sum(widths) + gap * (len(items) - 1)
        x = max(8, (WIN_W - total) // 2)
        y = WIN_H - FOOTER_H + 11
        for (ident, label, active), width in zip(items, widths):
            canvas.button(ident, label, x, y, width, 32, active)
            x += width + gap


def _score_label(score: int | str) -> str:
    if score == "X":
        return "X"
    if score == 0:
        return "мимо"
    return str(score)


def _score_color(score: int | str) -> tuple[int, int, int]:
    if score == "X":
        return GOLD
    if score == 0:
        return RED
    if isinstance(score, int) and score >= 9:
        return GREEN
    return TEXT


def _best_label(hits: list[Hit], target) -> str:
    if not hits:
        return "—"
    best = max(hits, key=lambda hit: (score_points(hit.score, target), hit.score == "X"))
    return "X" if best.score == "X" else str(best.score)


class CameraSource:
    def __init__(self, index: int):
        self.cap = _open_camera(index)
        self.name = f"Камера {index}"
        if self.cap is None:
            raise RuntimeError(f"Камера {index} не открылась")

    def read(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return False, None
        return True, downscale(frame)

    def release(self) -> None:
        self.cap.release()


class VideoSource:
    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        self.name = Path(path).name
        self.last = None
        if not self.cap.isOpened():
            raise RuntimeError(f"Не открывается видео: {path}")

    def read(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return False, self.last
        self.last = downscale(frame)
        return True, self.last

    def release(self) -> None:
        self.cap.release()


class DemoSource:
    def __init__(self, world: DemoWorld):
        self.world = world
        self.name = "Демо"

    def read(self) -> tuple[bool, np.ndarray]:
        return True, self.world.frame()

    def release(self) -> None:
        return None


def _open_camera(index: int):
    backends = [cv2.CAP_DSHOW] if sys.platform == "win32" else [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            cap.release()
            continue
        ok = False
        for _ in range(8):
            ok, frame = cap.read()
            if ok and frame is not None:
                break
        if ok:
            return cap
        cap.release()
    return None


def _print_help() -> None:
    print("ShotTracker — попадания с камеры")
    print("  Демо:      python app.py --demo")
    print("  Камера:    python app.py")
    print("  Видео:     python app.py --source запись.mp4")
    print("  Проверка:  python app.py --self-test")
    print()
    print("Камера должна стоять неподвижно. Свет ровный.")
    print("«Центр» — клик в середину мишени, затем по краю зоны 1.")
    print("«4 угла» — если камера смотрит под углом: углы листа, потом центр и край.")
    print("«Эталон» — текущая картинка без новых дыр. Дальше считаются только новые отверстия.")
    print("Чёрное яблоко при свете спереди почти не видно: поставьте свет за бумагой.")
    print("Горячие клавиши работают в английской раскладке. Кнопки — в любой.")
    print("Пробел в демо добавляет выстрел. Клик по схеме — ручное попадание.")


def main(argv: list[str] | None = None) -> None:
    _console_utf8()
    parser = argparse.ArgumentParser(description="ShotTracker — фиксация попаданий по мишени")
    parser.add_argument("--demo", action="store_true", help="Синтетическая мишень без камеры")
    parser.add_argument("--source", default="0", help="Индекс камеры, файл или URL")
    parser.add_argument("--self-test", action="store_true", help="Проверка детектора без окна")
    parser.add_argument("--no-sound", action="store_true", help="Без звука попадания")
    args = parser.parse_args(argv)
    if args.self_test:
        from shottracker.selftest import run
        sys.exit(run())

    world = None
    source = None
    if args.demo:
        world = DemoWorld()
        source = DemoSource(world)
    elif args.source.isdigit():
        try:
            source = CameraSource(int(args.source))
        except RuntimeError as exc:
            print(exc)
            print("Камера не найдена, запускаю демо. Для своей камеры: python app.py --source 1")
            world = DemoWorld()
            source = DemoSource(world)
    else:
        source = VideoSource(args.source)

    app = ShotApp(world=world, sound=not args.no_sound, source_name=getattr(source, "name", "Камера"))
    _print_help()
    rng = np.random.default_rng()
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, app.on_mouse)
    try:
        while True:
            if (
                app.world is not None
                and app.analysis
                and app.mode == "live"
                and app.detector.reference is not None
            ):
                now = time.time()
                if app._next_shot is None:
                    app._next_shot = now + 1.3
                elif now >= app._next_shot:
                    app.world.add_random_hole()
                    app._next_shot = now + float(rng.uniform(3.2, 5.8))
            ok, frame = source.read()
            if frame is None:
                print("Нет кадра.")
                break
            if not ok:
                app.announce("Видео закончилось")
                app.analysis = False
            canvas = app.step_frame(frame)
            cv2.imshow(WIN, canvas)
            key = cv2.waitKey(15)
            if key != -1 and not app.on_key(key & 0xFF):
                break
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        source.release()
        cv2.destroyAllWindows()
