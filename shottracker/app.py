"""Окно Great Shot Target: камера, калибровка, попадания, очки и группа."""

from __future__ import annotations

import argparse
import fcntl
import os
import struct
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
    deviation_pct,
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

ROOT = Path(__file__).resolve().parent.parent
CALIB_PATH = ROOT / "calibration.json"
RESULTS = ROOT / "results"
WIN = "Great Shot Target"
WIN_W, WIN_H = 1280, 780
HEADER_H, FOOTER_H = 56, 74

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
        self.session_no = 1
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
        self.camera_lost = False
        self._find_camera = False
        self._camera_misses = 0
        self._analysis_held = False
        self._chrome: np.ndarray | None = None
        self._chrome_key: tuple | None = None
        self._chrome_patches: list[tuple[int, int, int, int, np.ndarray]] = []
        self.note = ""
        self.prev_note = ""
        self.carry_note = False
        self._note_edit = False
        self.record_enabled = False
        self._writer: cv2.VideoWriter | None = None
        self._video_path: Path | None = None
        self._video_size: tuple[int, int] | None = None
        self._video_frames = 0
        self._video_next = 0.0
        self._session_stamp = ""
        self.video_parts: list[Path] = []
        self.snaps: list[bytes] = []
        self._cleared_snaps: list[bytes] = []
        self.review_hits: list[Hit] = []
        self.review_snaps: list[bytes] = []
        self.review_index: int | None = None
        self.browsing = False
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
        if self.camera_lost:
            return "Камера пропала. Нажмите «Вебка»"
        if self.mode == "corners":
            return "Правый клик — отмена"
        if self.mode == "center":
            return "Кликните центр. Правый клик — отмена"
        if self.mode == "edge":
            return "Кликните край зоны 1. Правый клик — отмена"
        if self.browsing and self.review_index is not None:
            hits, _snaps = self._series()
            if hits and self.review_index > 0:
                return self._step_text(hits, self.review_index - 1)
            if hits:
                return "Шаги: до первого выстрела"
        if self.calib is None:
            return "Нажмите «Центр» и отметьте мишень"
        if not self.analysis:
            return "Нажмите «Старт»"
        result = self.last_result
        if result is None:
            return "Старт включён"
        if result.status == "ok" and result.watching:
            return "Вижу новое пятно, подтверждаю…"
        return {
            "warmup": "Снимаю эталон, не стреляйте секунду…",
            "ready": "Эталон снят. Можно стрелять.",
            "skip-shift": "Сильный сдвиг камеры — кадр пропущен",
            "skip-motion": "В кадре движение — кадр пропущен",
            "skip-noise": "Много пятен. Поднимите порог",
            "ok": "Набор идёт",
        }.get(result.status, "Набор идёт")

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
            self.announce("Эталон обновится после «Старт»")

    def toggle_analysis(self) -> None:
        if self.calib is None:
            self.announce("Сначала откалибруйте мишень")
            return
        self.analysis = not self.analysis
        if self.analysis:
            self.browsing = False
            self.review_index = None
            self.announce("Старт")
        else:
            self._close_writer()
            self.announce("Пауза")

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

    def note_camera_lost(self) -> None:
        self._camera_misses += 1
        if self.camera_lost or self._camera_misses < 8:
            return
        self.camera_lost = True
        self._analysis_held = self.analysis
        self.analysis = False
        self.announce("Камера пропала. Нажмите «Вебка»", 6)

    def note_camera_back(self) -> None:
        self._camera_misses = 0
        if not self.camera_lost:
            return
        self.camera_lost = False
        if self._analysis_held:
            self.analysis = True
            self.detector.reset()
            self._next_shot = None
        self.announce("Камера снова в кадре", 3)

    def request_camera_search(self) -> None:
        self._find_camera = True
        self.announce("Ищу вебку…", 2)

    def use_camera(self, source: CameraSource, *, from_demo: bool) -> None:
        restore = self.camera_lost and self._analysis_held
        self.camera_lost = False
        self._camera_misses = 0
        self._analysis_held = False
        self.source_name = source.name
        self.detector.reset()
        self._next_shot = None
        if from_demo:
            self.world = None
            self.analysis = False
            self.mode = "live"
            self.pending_corners = []
            self.pending_center = None
            self.calib = None
            loaded = load_calibration(CALIB_PATH)
            if loaded:
                self.calib, key = loaded
                if key in TARGETS:
                    self.target_key = key
        elif restore:
            self.analysis = True
        self.announce(f"{source.name} найдена", 4)

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
            self.snaps = list(self._cleared_snaps)
            self._cleared = []
            self._cleared_snaps = []
            self.announce("Сброс отменён")
            return
        if not self.hits:
            self.announce("Нечего отменять")
            return
        self.hits.pop()
        if len(self.snaps) > len(self.hits):
            self.snaps.pop()
        if self.review_index is not None:
            self.review_index = min(self.review_index, len(self.hits))
        self.announce("Последний выстрел убран")

    def clear_hits(self) -> None:
        if not self.hits:
            self.announce("Список уже пуст")
            return
        self._cleared = list(self.hits)
        self._cleared_snaps = list(self.snaps)
        self.hits = []
        self.snaps = []
        self.review_index = None
        self.browsing = False
        self.announce("Попадания очищены. «Отмена» вернёт их")

    def export(self) -> None:
        if not self.hits:
            self.announce("Нет данных для экспорта")
            return
        RESULTS.mkdir(exist_ok=True)
        path = RESULTS / f"shottracker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        target = TARGETS[self.target_key]
        path.write_text(
            export_text(self.hits, target, group_report(self.hits), self.note, [p.name for p in self.video_parts]),
            encoding="utf-8",
        )
        print(f"Сохранено: {path}")
        self.announce(f"Сохранено {path.name}")

    def next_session(self) -> None:
        if not self.hits:
            self.announce("В этой сессии нет попаданий")
            return
        RESULTS.mkdir(exist_ok=True)
        self._close_writer()
        stamp = self._session_stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        path = RESULTS / f"session_{stamp}.png"
        image = self._session_card()
        if image is None or not cv2.imwrite(str(path), image):
            self.announce("Не удалось сохранить схему")
            return
        target = TARGETS[self.target_key]
        text_path = RESULTS / f"session_{stamp}.txt"
        videos = [item.name for item in self.video_parts]
        text_path.write_text(
            export_text(self.hits, target, group_report(self.hits), self.note, videos),
            encoding="utf-8",
        )
        self.review_hits = list(self.hits)
        self.review_snaps = list(self.snaps)
        self.hits = []
        self.snaps = []
        self._cleared = []
        self._cleared_snaps = []
        self.highlight = None
        self.prev_note = self.note
        if not self.carry_note:
            self.note = ""
        self.video_parts = []
        self._session_stamp = ""
        self.session_no += 1
        self.analysis = False
        self.browsing = True
        self.review_index = len(self.review_hits)
        print(f"Сессия сохранена: {path}")
        extra = f", видео {len(videos)}" if videos else ""
        kept = " Заметка перенесена." if self.carry_note and self.note else ""
        self.announce(f"Схема {path.name}{extra}.{kept} Шаги: ◀ ▶", 6)

    def _series(self) -> tuple[list[Hit], list[bytes]]:
        if self.browsing and not self.hits and self.review_hits:
            return self.review_hits, self.review_snaps
        return self.hits, self.snaps

    def _shown_hits(self) -> list[Hit]:
        hits, _snaps = self._series()
        if self.review_index is None:
            return hits
        return hits[:self.review_index]

    def _step_text(self, hits: list[Hit], index: int) -> str:
        hit = hits[index]
        pct = deviation_pct(hit)
        tail = ""
        if index > 0:
            prev = deviation_pct(hits[index - 1])
            if pct + 0.5 < prev:
                tail = " · лучше предыдущего"
            elif pct > prev + 0.5:
                tail = " · хуже предыдущего"
        return f"Шаг {index + 1}/{len(hits)} · отклонение {pct:.0f}%{tail}"

    def _announce_step(self) -> None:
        hits, _snaps = self._series()
        if not hits or self.review_index is None:
            return
        if self.review_index <= 0:
            self.announce("Шаги: до первого выстрела", 4)
            self.highlight = None
            return
        self.highlight = self.review_index - 1
        self.highlight_until = time.time() + 3600
        self.announce(self._step_text(hits, self.review_index - 1), 4)

    def step_review(self, delta: int) -> None:
        if self.analysis:
            self.analysis = False
            self._close_writer()
        hits, _snaps = (self.review_hits, self.review_snaps) if not self.hits and self.review_hits else (self.hits, self.snaps)
        if not hits:
            self.announce("Нет выстрелов для шагов")
            return
        if not self.hits and self.review_hits:
            self.browsing = True
        if self.hits:
            self.browsing = True
        if self.review_index is None:
            self.review_index = len(hits)
        self.review_index = max(0, min(len(hits), self.review_index + delta))
        self._announce_step()

    def jump_review(self, index: int) -> None:
        hits, _snaps = self._series()
        if index < 0 or index >= len(hits):
            return
        if self.analysis:
            self.analysis = False
            self._close_writer()
        self.browsing = True
        self.review_index = index + 1
        self._announce_step()

    def toggle_record(self) -> None:
        self.record_enabled = not self.record_enabled
        if not self.record_enabled:
            self._close_writer()
        self.announce("Запись сессии включена" if self.record_enabled else "Запись выключена")

    def toggle_carry_note(self) -> None:
        self.carry_note = not self.carry_note
        self.announce("Заметка перейдёт на следующую сессию" if self.carry_note else "Следующая сессия начнётся без заметки")

    def copy_previous_note(self) -> None:
        if not self.prev_note:
            self.announce("Прошлая заметка пуста")
            return
        self.note = self.prev_note
        self.announce("Заметка скопирована с прошлой сессии")

    def request_note_edit(self) -> None:
        self._note_edit = True

    def edit_note(self) -> None:
        updated = _ask_note(self.note)
        if updated is None:
            return
        self.note = updated
        self.announce("Заметка сохранена" if updated else "Заметка очищена")

    def _fill_snaps(self, image: np.ndarray) -> None:
        while len(self.snaps) < len(self.hits):
            snap = image
            if snap.shape[1] > 640:
                height = max(1, int(snap.shape[0] * 640 / snap.shape[1]))
                snap = cv2.resize(snap, (640, height), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", snap, [cv2.IMWRITE_JPEG_QUALITY, 80])
            self.snaps.append(buf.tobytes() if ok else b"")

    def _store_snap(self) -> None:
        image = self.last_view.image if self.last_view is not None else self.raw_frame
        if image is None:
            self.snaps.append(b"")
            return
        self._fill_snaps(image)

    def _review_snap(self) -> np.ndarray | None:
        if not self.browsing or self.review_index is None or self.review_index <= 0:
            return None
        _hits, snaps = self._series()
        index = self.review_index - 1
        if index >= len(snaps) or not snaps[index]:
            return None
        frame = cv2.imdecode(np.frombuffer(snaps[index], np.uint8), cv2.IMREAD_COLOR)
        return frame

    def _close_writer(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._video_path is not None:
            self.video_parts.append(self._video_path)
            self._video_path = None
        self._video_size = None

    def _write_video(self, image: np.ndarray) -> None:
        if not self.record_enabled or not self.analysis or self.browsing:
            return
        now = time.time()
        if self._writer is not None and now < self._video_next:
            return
        self._video_next = now + (1.0 / 12.0)
        if self._writer is None:
            RESULTS.mkdir(exist_ok=True)
            if not self._session_stamp:
                self._session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            part = len(self.video_parts) + 1
            name = f"session_{self._session_stamp}.avi" if part == 1 else f"session_{self._session_stamp}_{part}.avi"
            height, width = image.shape[:2]
            width -= width % 2
            height -= height % 2
            if width < 2 or height < 2:
                return
            path = RESULTS / name
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 12.0, (width, height))
            if not writer.isOpened():
                self.record_enabled = False
                self.announce("Не удалось начать запись видео")
                return
            self._writer = writer
            self._video_path = path
            self._video_size = (width, height)
            self._video_frames = 0
        assert self._video_size is not None
        width, height = self._video_size
        frame = image
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        self._writer.write(frame)
        self._video_frames += 1

    def _session_card(self) -> np.ndarray:
        target = TARGETS[self.target_key]
        group = group_report(self.hits)
        width, height = 880, 1140 if self.note.strip() else 980
        canvas = Canvas(width, height)
        canvas.fill(0, 0, width, height, BG)
        canvas.text(40, 28, "GREAT SHOT TARGET", 16, GOLD, True)
        canvas.text(40, 56, target.name, 28, TEXT, True)
        canvas.text(
            40, 100,
            f"Сессия {self.session_no}    {datetime.now().strftime('%d.%m.%Y  %H:%M')}",
            16, MUTED, False,
        )
        side = 640
        left = (width - side) // 2
        top = 148
        canvas.fill(left, top, side, side, (8, 8, 8))
        cx = left + side / 2.0
        cy = top + side / 2.0
        radius = side * 0.46
        paint_rings(canvas.img, cx, cy, radius, target)
        draw_hits(canvas.img, cx, cy, radius, self.hits, None, show_poi=True)
        canvas.rect(left, top, side, side, GOLD, 1)
        total = sum(score_points(hit.score, target) for hit in self.hits)
        count = len(self.hits)
        average = f"{total / count:.1f}" if count else "—"
        lines = [
            f"Выстрелов   {count}",
            f"Сумма   {total}",
            f"Среднее   {average}",
            f"Лучший   {_best_label(self.hits, target)}",
        ]
        if group:
            lines.extend((
                f"Разброс   {group['spread']:.3f}",
                f"СТП   {group['avg_x']:+.2f} / {group['avg_y']:+.2f}",
                f"Смещение   {group['direction']}",
            ))
        if devs := [deviation_pct(hit) for hit in self.hits]:
            lines.append(f"Отклонение   {sum(devs) / len(devs):.0f}%")
        y = top + side + 22
        for index, line in enumerate(lines):
            column = index // 4
            row = index % 4
            canvas.text(40 + column * 420, y + row * 28, line, 18, TEXT, False)
        if self.note.strip():
            note_y = y + 4 * 28 + 10
            canvas.text(40, note_y, "ЗАМЕТКА", 13, GOLD, True)
            for index, line in enumerate(_wrap_note(self.note)[:5]):
                canvas.text(40, note_y + 22 + index * 22, line, 16, TEXT, False)
        return canvas.finish()

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
        elif ident == "session":
            self.next_session()
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
        elif ident == "webcam":
            self.request_camera_search()
        elif ident == "record":
            self.toggle_record()
        elif ident == "note":
            self.request_note_edit()
        elif ident == "note-copy":
            self.copy_previous_note()
        elif ident == "note-carry":
            self.toggle_carry_note()
        elif ident == "step-":
            self.step_review(-1)
        elif ident == "step+":
            self.step_review(1)

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
            ord("n"): self.next_session,
            ord("N"): self.next_session,
            ord("e"): self.export,
            ord("E"): self.export,
            ord("x"): self.clear_hits,
            ord("X"): self.clear_hits,
            ord("z"): self.undo,
            ord("Z"): self.undo,
            ord(" "): self.demo_shot,
            ord("m"): self._toggle_mask,
            ord("M"): self._toggle_mask,
            ord("v"): self.request_camera_search,
            ord("V"): self.request_camera_search,
            ord(","): lambda: self.step_review(-1),
            ord("."): lambda: self.step_review(1),
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
                self.jump_review(index)
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
        if self.browsing:
            return
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
        self.hits.append(Hit(norm_x, norm_y, score, datetime.now(), source, self._video_frames))
        self.browsing = False
        self.review_index = None
        label = "X" if score == "X" else ("мимо" if score == 0 else str(score))
        pct = deviation_pct(self.hits[-1])
        tail = ""
        earlier = [deviation_pct(hit) for hit in self.hits[:-1]]
        if earlier:
            if pct + 0.5 < earlier[-1]:
                tail = " · лучше предыдущего"
            elif pct > earlier[-1] + 0.5:
                tail = " · хуже предыдущего"
            if pct + 0.5 < min(earlier):
                tail = " · лучший в серии"
        self.announce(f"Выстрел {len(self.hits)} · {label} · отклонение {pct:.0f}%{tail}", 2.8)
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
        snap = self._review_snap()
        if snap is not None and self.mode == "live":
            return self._render(snap, "raw", None, None)
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
            self._fill_snaps(preview)
            self._write_video(preview)
            space = "work"
        else:
            space = "raw"
            if self.mode == "corners" or (self.mode == "edge" and self.pending_center):
                preview = frame.copy()
                self._draw_pending(preview, space)
            else:
                preview = frame
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

    def _chrome_sig(self, preview: np.ndarray, space: str) -> tuple:
        now = time.time()
        hits = tuple((hit.norm_x, hit.norm_y, hit.score, hit.source) for hit in self.hits)
        return (
            preview.shape,
            space,
            self.mode,
            self.pending_space,
            len(self.pending_corners),
            self.analysis,
            self.camera_lost,
            self.target_key,
            self.session_no,
            self.detector.sensitivity,
            self.size_scale,
            self.show_mask,
            None if self.calib is None else (self.calib.space, self.calib.outer_r),
            self.overlay_message(),
            self._pill_text(),
            hits,
            now < self.flash_until,
            self.highlight if now < self.highlight_until else None,
            self.note,
            self.carry_note,
            self.record_enabled,
            self.browsing,
            self.review_index,
            len(self.review_hits),
            self.source_name,
        )

    def _render(self, preview: np.ndarray, space: str, view, result) -> np.ndarray:
        cam = (12, HEADER_H + 8, 742, WIN_H - HEADER_H - FOOTER_H - 16)
        sig = self._chrome_sig(preview, space)
        if self._chrome is None or sig != self._chrome_key:
            canvas = Canvas(WIN_W, WIN_H)
            self._header(canvas)
            self._camera(canvas, cam, preview, space, result)
            panel = (cam[0] + cam[2] + 10, cam[1], WIN_W - (cam[0] + cam[2] + 20), cam[3])
            self._side(canvas, panel, view)
            self._footer(canvas)
            self.buttons = list(canvas.buttons)
            self._chrome = canvas.finish()
            self._chrome_key = sig
            self._chrome_patches = self._video_patches()
        out = self._chrome.copy()
        ox, oy, nw, nh, scale = blit_contain(out, preview, cam)
        self.view_geom = ViewGeom(ox, oy, nw, nh, scale, preview.shape[1], preview.shape[0], space)
        for y1, y2, x1, x2, patch in self._chrome_patches:
            out[y1:y2, x1:x2] = patch
        self._rubber_band(out, ox, oy, scale)
        if self.show_mask and result is not None and result.mask is not None and nw > 180:
            thumb = cv2.cvtColor(result.mask, cv2.COLOR_GRAY2BGR)
            thumb = cv2.resize(thumb, (150, 150), interpolation=cv2.INTER_NEAREST)
            tx, ty = ox + nw - 158, oy + nh - 158
            out[ty:ty + 150, tx:tx + 150] = thumb
            cv2.rectangle(out, (tx, ty), (tx + 149, ty + 149), GOLD, 1)
        return out

    def _video_patches(self) -> list[tuple[int, int, int, int, np.ndarray]]:
        geom = self.view_geom
        chrome = self._chrome
        if geom is None or chrome is None or geom.nw <= 0 or geom.nh <= 0:
            return []
        strips: list[tuple[int, int, int, int]] = []
        if self.mode != "live":
            strips.append((geom.oy, min(geom.oy + 34, geom.oy + geom.nh), geom.ox, geom.ox + geom.nw))
        elif self.analysis:
            strips.append((geom.oy, min(geom.oy + 24, geom.oy + geom.nh), geom.ox, min(geom.ox + 88, geom.ox + geom.nw)))
        patches = []
        height, width = chrome.shape[:2]
        for y1, y2, x1, x2 in strips:
            y1, y2 = max(0, y1), min(height, y2)
            x1, x2 = max(0, x1), min(width, x2)
            if y2 > y1 and x2 > x1:
                patches.append((y1, y2, x1, x2, chrome[y1:y2, x1:x2].copy()))
        return patches

    def _header(self, canvas: Canvas) -> None:
        canvas.fill(0, 0, WIN_W, HEADER_H, PANEL)
        cv2.circle(canvas.img, (28, 28), 12, GOLD, 2, lineType=cv2.LINE_AA)
        cv2.circle(canvas.img, (28, 28), 4, GOLD, -1, lineType=cv2.LINE_AA)
        canvas.text(48, 16, "GREAT SHOT", 20, TEXT, True)
        shot_w, _, _, _ = text_metrics("GREAT SHOT", 20, True)
        canvas.text(48 + shot_w + 8, 16, "TARGET", 20, GOLD, True)
        target = TARGETS[self.target_key]
        hits = self._series()[0]
        n = len(hits)
        total = sum(score_points(hit.score, target) for hit in hits)
        avg = f"{total / n:.1f}" if n else "—"
        best = _best_label(hits, target)
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
            return f"{self.source_name} · сессия {self.session_no}"
        return f"{self.source_name} · пауза"

    def _camera(self, canvas: Canvas, box: tuple[int, int, int, int], preview: np.ndarray, space: str, result) -> None:
        canvas.fill(*box, (0, 0, 0))
        ox, oy, nw, nh, scale = blit_contain(canvas.img, preview, box)
        self.view_geom = ViewGeom(ox, oy, nw, nh, scale, preview.shape[1], preview.shape[0], space)
        if self.mode != "live" and nw > 0:
            canvas.fill(ox, oy, nw, 34, GOLD)
            canvas.text(ox + 12, oy + 8, self.calib_banner(), 15, INK, True)
            self._rubber_band(canvas.img, ox, oy, scale)
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

    def _rubber_band(self, image: np.ndarray, ox: int, oy: int, scale: float) -> None:
        if self.mode != "edge" or self.pending_center is None or self.mouse is None or self.view_geom is None:
            return
        if self.view_geom.space != self.pending_space:
            return
        sx = ox + self.pending_center[0] * scale
        sy = oy + self.pending_center[1] * scale
        radius = hypot(self.mouse[0] - sx, self.mouse[1] - sy)
        if radius < 2:
            return
        cv2.circle(image, (int(sx), int(sy)), int(radius), GOLD, 1, lineType=cv2.LINE_AA)

    def _side(self, canvas: Canvas, panel: tuple[int, int, int, int], view) -> None:
        x, y, w, h = panel
        canvas.round_box(x, y, w, h, PANEL, BORDER, 10)
        cursor = y + 10
        canvas.button("target", TARGETS[self.target_key].name, x + 12, cursor, w - 24, 30, False)
        cursor += 36
        canvas.text(x + 14, cursor, self.overlay_message(), 13, GOLD, False)
        cursor += 22
        self._note_box(canvas, x + 12, cursor, w - 24)
        cursor += 62
        group_h = 88
        slider_h = 64
        rest = h - (cursor - y) - group_h - slider_h - 16
        schem_h = int(max(160, min(rest * 0.62, 250)))
        schem = (x + 12, cursor, w - 24, schem_h)
        self._schematic(canvas, schem)
        cursor += schem_h + 8
        self._group_box(canvas, x + 12, cursor, w - 24, group_h)
        cursor += group_h + 8
        self._log(canvas, x + 12, cursor, w - 24, max(48, y + h - slider_h - 8 - cursor))
        self._sliders(canvas, x + 12, y + h - slider_h, w - 24)

    def _note_box(self, canvas: Canvas, x: int, y: int, w: int) -> None:
        preview = " ".join(self.note.split())
        if preview:
            color = TEXT
        else:
            preview = "Погода, ветер — кнопка «Заметка»"
            color = MUTED
        if len(preview) > 48:
            preview = preview[:47] + "…"
        canvas.text(x, y, preview, 13, color, False)
        canvas.button("note", "Заметка", x, y + 20, 96, 26, False)
        canvas.button("note-copy", "Из прошлой", x + 104, y + 20, 124, 26, False)
        carry_w = min(100, max(80, w - 236))
        canvas.button("note-carry", "Перенос", x + 236, y + 20, carry_w, 26, self.carry_note)

    def _schematic(self, canvas: Canvas, box: tuple[int, int, int, int]) -> None:
        x, y, w, h = box
        canvas.fill(x, y, w, h, BG)
        cx = x + w / 2.0
        cy = y + h / 2.0
        radius = min(w, h) * 0.46
        self.schem_geom = (cx, cy, radius, box)
        hits = self._shown_hits()
        if self.review_index is not None and self.review_index > 0:
            highlight: int | None = self.review_index - 1
        else:
            highlight = self.highlight if time.time() < self.highlight_until else None
        if highlight is not None and highlight >= len(hits):
            highlight = None
        paint_rings(canvas.img, cx, cy, radius, TARGETS[self.target_key])
        draw_hits(canvas.img, cx, cy, radius, hits, highlight, show_poi=True)

    def _group_box(self, canvas: Canvas, x: int, y: int, w: int, h: int) -> None:
        canvas.round_box(x, y, w, h, PANEL2, BORDER, 8)
        canvas.text(x + 12, y + 6, "ГРУППА", 11, MUTED, True)
        report = group_report(self._shown_hits())
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
        xcount = sum(1 for hit in self._shown_hits() if hit.score == "X")
        canvas.text(x + w - 78, y + 6, f"X {xcount}", 12, GOLD, True)

    def _log(self, canvas: Canvas, x: int, y: int, w: int, h: int) -> None:
        self.hit_rows = []
        if h < 24:
            return
        hits, _snaps = self._series()
        canvas.text(x, y, f"СЕССИЯ {self.session_no}", 11, MUTED, True)
        canvas.button("step-", "◀", x + w - 64, y - 4, 28, 22, False)
        canvas.button("step+", "▶", x + w - 32, y - 4, 28, 22, self.browsing)
        row_h = 22
        slots = max(0, (h - 18) // row_h)
        if slots <= 0 or not hits:
            if not hits:
                canvas.text(x, y + 22, "Пока пусто — клик по мишени тоже считается", 13, MUTED, False)
            return
        visible = list(enumerate(hits))[-slots:]
        current = self.review_index - 1 if self.review_index else None
        highlight = current if current is not None else (self.highlight if time.time() < self.highlight_until else None)
        for row, (index, hit) in enumerate(reversed(visible)):
            yy = y + 18 + row * row_h
            if highlight == index:
                canvas.fill(x, yy - 1, w, row_h, PANEL2)
            self.hit_rows.append(((x, yy - 1, w, row_h), index))
            canvas.text(x, yy, f"{index + 1:02d}", 13, MUTED, True)
            canvas.text(x + 36, yy, _score_label(hit.score), 14, _score_color(hit.score), True)
            pct = deviation_pct(hit)
            tone = GREEN if pct < 15 else RED if pct > 50 else TEXT
            canvas.text(x + 78, yy, f"{pct:.0f}%", 14, tone, True)
            canvas.text(x + 140, yy, hit.stamp.strftime("%H:%M:%S"), 13, MUTED, False)

    def _sliders(self, canvas: Canvas, x: int, y: int, w: int) -> None:
        canvas.text(x, y + 4, f"Порог {self.detector.sensitivity}", 13, TEXT, True)
        canvas.button("sens-", "−", x + 108, y, 28, 26, False)
        canvas.button("sens+", "+", x + 140, y, 28, 26, False)
        canvas.text(x, y + 36, f"Пятно ×{self.size_scale:.1f}", 13, TEXT, True)
        canvas.button("size-", "−", x + 108, y + 32, 28, 26, False)
        canvas.button("size+", "+", x + 140, y + 32, 28, 26, False)
        canvas.button("mask", "Маска", x + w - 178, y + 32, 84, 26, self.show_mask)
        canvas.button("record", "Запись", x + w - 86, y + 32, 86, 26, self.record_enabled)

    def _footer(self, canvas: Canvas) -> None:
        canvas.fill(0, WIN_H - FOOTER_H, WIN_W, FOOTER_H, PANEL)
        start_label = "Стоп" if self.analysis else "Старт"
        items = [
            ("center", "Центр", self.mode in ("center", "edge") and self.pending_space == "raw"),
            ("corners", "4 угла", self.mode == "corners" or self.pending_space == "warp" and self.mode != "live"),
            ("auto", "Авто", False),
            ("webcam", "Вебка", self.camera_lost),
            ("reference", "Эталон", False),
            ("analysis", start_label, self.analysis),
            ("session", "Новая", False),
            ("export", "Экспорт", False),
            ("clear", "Сброс", False),
            ("undo", "Отмена", False),
        ]
        if self.world is not None:
            items.append(("shot", "Выстрел", False))
        gap = 8
        widths = []
        for ident, label, _active in items:
            size = 20 if ident == "analysis" else 14
            extra = 56 if ident == "analysis" else 28
            tw, _, _, _ = text_metrics(label, size, True)
            widths.append(max(tw + extra, 156 if ident == "analysis" else 0))
        total = sum(widths) + gap * (len(items) - 1)
        x = max(8, (WIN_W - total) // 2)
        small_h = 32
        start_h = 52
        y_small = WIN_H - FOOTER_H + (FOOTER_H - small_h) // 2
        y_start = WIN_H - FOOTER_H + (FOOTER_H - start_h) // 2
        for (ident, label, active), width in zip(items, widths):
            if ident == "analysis":
                canvas.button(ident, label, x, y_start, width, start_h, active, size=20, radius=12)
            else:
                canvas.button(ident, label, x, y_small, width, small_h, active)
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
        self.index = index
        self.cap = _open_camera(index)
        self.name = f"Камера {index}"
        self.last: np.ndarray | None = None
        if self.cap is None:
            raise RuntimeError(f"Камера {index} не открылась")

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.cap is None or not self.cap.isOpened():
            return False, self.last
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return False, self.last
        self.last = downscale(frame)
        return True, self.last

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


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


def _capture_indexes() -> list[int]:
    """Индексы V4L2, с которых реально идёт картинка. Служебный узел C920 пропускается."""
    if sys.platform == "win32" or not os.path.isdir("/sys/class/video4linux"):
        return list(range(6))
    indexes = []
    for name in sorted(os.listdir("/sys/class/video4linux")):
        if not name.startswith("video") or not name[5:].isdigit():
            continue
        if _v4l2_capture(f"/dev/{name}"):
            indexes.append(int(name[5:]))
    return indexes


def _v4l2_capture(path: str) -> bool:
    video = 0x00000001
    mplane = 0x00001000
    device_caps_flag = 0x80000000
    try:
        fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
    except OSError:
        return False
    try:
        buf = bytearray(104)
        fcntl.ioctl(fd, 0x80685600, buf)
    except OSError:
        return False
    finally:
        os.close(fd)
    capabilities, device_caps = struct.unpack_from("<II", buf, 84)
    caps = device_caps if capabilities & device_caps_flag else capabilities
    return bool(caps & (video | mplane))


def open_first_camera(prefer: int | None = None) -> CameraSource | None:
    indexes = _capture_indexes()
    if prefer is not None:
        indexes = [prefer] + [index for index in indexes if index != prefer]
    for index in indexes:
        try:
            return CameraSource(index)
        except RuntimeError:
            continue
    return None


def search_camera(source, app: ShotApp):
    prefer = source.index if isinstance(source, CameraSource) else None
    from_demo = isinstance(source, DemoSource)
    previous = source if isinstance(source, CameraSource) else None
    if previous is not None:
        previous.release()
    found = open_first_camera(prefer)
    if found is None:
        app.announce("Вебка не найдена", 4)
        if previous is not None:
            if not app.camera_lost:
                app._analysis_held = app.analysis
                app.analysis = False
            app.camera_lost = True
        return previous if previous is not None else source
    if previous is None:
        source.release()
    app.use_camera(found, from_demo=from_demo)
    return found


def _open_camera(index: int):
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2
    # Сначала MJPG: C920 в YUYV отдаёт 720p слишком редко, и картинка отстаёт.
    for mjpg in (True, False):
        cap = cv2.VideoCapture(index, backend)
        if mjpg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            cap.release()
            continue
        ok = False
        frame = None
        for _ in range(8):
            ok, frame = cap.read()
            if ok and frame is not None:
                break
        if not ok or frame is None:
            cap.release()
            continue
        if mjpg:
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            code = "".join(chr((fourcc >> (8 * i)) & 255) for i in range(4))
            if "MJPG" not in code.upper():
                cap.release()
                continue
        return cap
    return None


def _wrap_note(text: str, width: int = 48) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        piece = word if not current else f"{current} {word}"
        if len(piece) <= width:
            current = piece
            continue
        if current:
            lines.append(current)
        current = word
    if current:
        lines.append(current)
    return lines


def _ask_note(current: str) -> str | None:
    try:
        import tkinter as tk
    except ImportError:
        print("Нет tkinter — заметку не открыть.")
        return None
    result: dict[str, str | None] = {"text": None}
    root = tk.Tk()
    root.title("Заметка сессии")
    root.configure(bg="#161616")
    root.geometry("520x280")
    try:
        root.attributes("-topmost", True)
    except tk.TclError:
        pass
    tk.Label(
        root,
        text="Погода, ветер и прочее",
        bg="#161616",
        fg="#e8e0d0",
        font=("DejaVu Sans", 12),
    ).pack(anchor="w", padx=16, pady=(14, 6))
    box = tk.Text(
        root,
        height=8,
        wrap="word",
        bg="#101010",
        fg="#f2efe8",
        insertbackground="#f2efe8",
        font=("DejaVu Sans", 12),
    )
    box.pack(fill="both", expand=True, padx=16)
    box.insert("1.0", current)

    def save(_event=None) -> None:
        result["text"] = box.get("1.0", "end").strip()
        root.destroy()

    def cancel(_event=None) -> None:
        root.destroy()

    row = tk.Frame(root, bg="#161616")
    row.pack(fill="x", padx=16, pady=12)
    tk.Button(row, text="Сохранить", command=save).pack(side="left")
    tk.Button(row, text="Отмена", command=cancel).pack(side="left", padx=8)
    root.bind("<Escape>", cancel)
    root.protocol("WM_DELETE_WINDOW", cancel)
    root.mainloop()
    return result["text"]


def _print_help() -> None:
    print("Great Shot Target — попадания с камеры")
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
    print("«Вебка» (V) — заново найти камеру, если она пропала.")
    print("«Старт» — начать набор. «Новая» (N) — сохранить схему сессии и очистить список.")
    print("«Запись» — писать видео сессии. «◀ ▶» или «,» «.» — шаги по выстрелам.")
    print("«Заметка» — погода и ветер. «Перенос» копирует её на следующую сессию, «Из прошлой» — обратно.")


def main(argv: list[str] | None = None) -> None:
    _console_utf8()
    parser = argparse.ArgumentParser(description="Great Shot Target — фиксация попаданий по мишени")
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
        source = open_first_camera(int(args.source))
        if source is None:
            print(f"Камера {args.source} не открылась")
            print("Камера не найдена, запускаю демо. Нажмите «Вебка», когда она появится.")
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
            if app._find_camera:
                app._find_camera = False
                source = search_camera(source, app)
            if app._note_edit:
                app._note_edit = False
                app.edit_note()
            ok, frame = source.read()
            if isinstance(source, CameraSource):
                if ok and frame is not None:
                    app.note_camera_back()
                else:
                    app.note_camera_lost()
                    if frame is None:
                        frame = np.zeros((720, 1280, 3), np.uint8)
            elif frame is None:
                print("Нет кадра.")
                break
            elif not ok:
                app.announce("Видео закончилось")
                app.analysis = False
            canvas = app.step_frame(frame)
            cv2.imshow(WIN, canvas)
            key = cv2.waitKey(1)
            if key != -1 and not app.on_key(key & 0xFF):
                break
            # 0 — окно закрыли. -1 — бэкенд не сообщает видимость (Xvfb в контейнере).
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) == 0:
                break
    finally:
        source.release()
        cv2.destroyAllWindows()
