"""Проверка очков, калибровки и детектора без окна."""

from __future__ import annotations

import traceback
from math import hypot

import cv2
import numpy as np

from shottracker.app import RESULTS, ShotApp
from shottracker.demo import DemoWorld
from shottracker.detector import alignment_delta
from shottracker.geometry import Calibration, find_circle, translate, work_view
from shottracker.targets import TARGETS, group_report, score_at, score_points


def test_scores() -> None:
    pistol = TARGETS["pistol"]
    if score_at(0, 0, pistol) != "X":
        raise AssertionError(score_at(0, 0, pistol))
    if score_at(0.08, 0, pistol) != 10:
        raise AssertionError(score_at(0.08, 0, pistol))
    if score_at(0.2, 0, pistol) != 8:
        raise AssertionError(score_at(0.2, 0, pistol))
    if score_at(1.05, 0, pistol) != 0:
        raise AssertionError("край должен быть промахом")
    if score_points("X", pistol) != 10:
        raise AssertionError("X пистолета должен давать 10")
    if score_points("X", TARGETS["biathlon"]) != 5:
        raise AssertionError("X биатлона должен давать 5")
    report = group_report([])
    if report is not None:
        raise AssertionError("пустая группа")


def test_alignment() -> None:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (240, 240), np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    rolled = np.roll(np.roll(image, 7, axis=1), -4, axis=0)
    shift, response = cv2.phaseCorrelate(image.astype(np.float32), rolled.astype(np.float32))
    if abs(shift[0] - 7) > 0.75 or abs(shift[1] + 4) > 0.75:
        raise AssertionError(f"phaseCorrelate shift={shift} response={response}, ждали ~(7, -4)")
    dx, dy, resp = alignment_delta(image, rolled)
    fixed = translate(rolled, dx, dy)
    err = float(np.mean(np.abs(image[20:220, 20:220].astype(np.int16) - fixed[20:220, 20:220].astype(np.int16))))
    if err > 12:
        raise AssertionError(f"после выравнивания ошибка {err:.2f}, сдвиг {(dx, dy)}, response {resp}")


def test_geometry() -> None:
    width, height = 960, 720
    cx, cy, radius = 480.0, 360.0, 288.0
    image = np.zeros((height, width, 3), np.uint8)
    point = (int(round(cx + radius)), int(round(cy)))
    cv2.circle(image, point, 5, (255, 255, 255), -1)
    view = work_view(image, Calibration("raw", (cx, cy), radius, None, (width, height)))
    gray = cv2.cvtColor(view.image, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > 200)
    if len(xs) == 0:
        raise AssertionError("точка на краю зоны 1 пропала после выравнивания кадра")
    found = (float(xs.mean()), float(ys.mean()))
    expected = (view.cx + view.outer_r, view.cy)
    if hypot(found[0] - expected[0], found[1] - expected[1]) > 2.5:
        raise AssertionError(f"край зоны 1 уехал: {found}, ждали {expected}")


def _pump(app: ShotApp, world: DemoWorld, count: int) -> None:
    for _ in range(count):
        app.step_frame(world.frame())


def test_pipeline() -> None:
    world = DemoWorld(seed=2, noise=1.6)
    app = ShotApp(world=world, sound=False)
    _pump(app, world, 8)
    if app.detector.reference is None:
        raise AssertionError("эталон не снялся")
    if app.hits:
        raise AssertionError(f"ложные попадания на чистой мишени: {app.hits}")

    world.add_hole(world.center[0], world.center[1])
    _pump(app, world, 16)
    if len(app.hits) != 1:
        _dump(app, "pipeline-center")
        raise AssertionError(f"центр: ждали 1 попадание, есть {len(app.hits)} { _fmt(app) }")
    hit = app.hits[0]
    if hypot(hit.norm_x, hit.norm_y) > 0.06 or hit.score != "X":
        _dump(app, "pipeline-center")
        raise AssertionError(f"центр должен быть X, получили {hit.norm_x:.3f},{hit.norm_y:.3f} {hit.score}")

    world.add_hole(world.center[0] + world.outer_r * 0.55, world.center[1])
    _pump(app, world, 16)
    if len(app.hits) != 2:
        _dump(app, "pipeline-second")
        raise AssertionError(f"второе отверстие: {len(app.hits)} {_fmt(app)}")
    second = app.hits[1]
    if abs(second.norm_x - 0.55) > 0.07 or abs(second.norm_y) > 0.07:
        _dump(app, "pipeline-second")
        raise AssertionError(f"координата второй дыры {second.norm_x:.3f},{second.norm_y:.3f}")

    _pump(app, world, 10)
    if len(app.hits) != 2:
        _dump(app, "pipeline-dup")
        raise AssertionError("дыра посчиталась повторно")
    print("   hits:", _fmt(app))


def test_flash_and_motion() -> None:
    world = DemoWorld(seed=3, noise=1.5)
    app = ShotApp(world=world, sound=False)
    _pump(app, world, 8)
    for _ in range(3):
        frame = world.frame()
        frame = np.clip(frame.astype(np.int16) + 70, 0, 255).astype(np.uint8)
        app.step_frame(frame)
    for _ in range(2):
        frame = world.frame()
        cv2.circle(frame, (int(world.center[0] + 50), int(world.center[1] - 30)), 8, (250, 250, 250), -1)
        app.step_frame(frame)
    for _ in range(3):
        frame = world.frame()
        frame[40:460, 30:520] = 0
        app.step_frame(frame)
    _pump(app, world, 10)
    if app.hits:
        _dump(app, "flash")
        raise AssertionError(f"вспышка или движение засчитались: {_fmt(app)}")


def test_auto_circle() -> None:
    world = DemoWorld(seed=1, noise=0)
    found = find_circle(world.plain)
    if found is None:
        raise AssertionError("автокруг не нашёл мишень")
    _x, _y, radius = found
    if abs(radius - world.outer_r) / world.outer_r > 0.12:
        raise AssertionError(f"радиус автокруга {radius:.1f}, у мишени {world.outer_r:.1f}")


def save_preview() -> None:
    world = DemoWorld(seed=5, noise=1.4)
    app = ShotApp(world=world, sound=False)
    _pump(app, world, 8)
    world.add_hole(world.center[0], world.center[1])
    _pump(app, world, 14)
    world.add_hole(world.center[0] + world.outer_r * 0.42, world.center[1] - world.outer_r * 0.2)
    _pump(app, world, 14)
    world.add_hole(world.center[0] - world.outer_r * 0.15, world.center[1] + world.outer_r * 0.33)
    canvas = None
    for _ in range(14):
        canvas = app.step_frame(world.frame())
    if len(app.hits) < 3:
        _dump(app, "preview")
        raise AssertionError(f"для превью ждали 3 попадания, есть {len(app.hits)}")
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "preview.png"
    if canvas is None or not cv2.imwrite(str(path), canvas):
        raise AssertionError("не сохранился preview.png")
    print(f"   preview {path}")


def _fmt(app: ShotApp) -> list[tuple]:
    return [(round(hit.norm_x, 3), round(hit.norm_y, 3), hit.score) for hit in app.hits]


def _dump(app: ShotApp, name: str) -> None:
    RESULTS.mkdir(exist_ok=True)
    if app.last_view is not None:
        cv2.imwrite(str(RESULTS / f"debug-{name}.png"), app.last_view.image)
    if app.last_result is not None and app.last_result.mask is not None:
        cv2.imwrite(str(RESULTS / f"debug-{name}-mask.png"), app.last_result.mask)
    print(f"   debug saved for {name}, status={getattr(app.last_result, 'status', None)}")


def run() -> int:
    tests = [
        test_scores,
        test_alignment,
        test_geometry,
        test_auto_circle,
        test_pipeline,
        test_flash_and_motion,
    ]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  ok  {test.__name__}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL {test.__name__}: {exc}")
            traceback.print_exc()
    if failed == 0:
        try:
            save_preview()
            print("  ok  save_preview")
        except Exception as exc:
            failed += 1
            print(f"  FAIL save_preview: {exc}")
            traceback.print_exc()
    print("SELF-TEST", "PASSED" if failed == 0 else f"FAILED ({failed})")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(run())
