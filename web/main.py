"""Страницы, вход и живой разбор кадров."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import Counter
from contextlib import asynccontextmanager
from math import hypot
from pathlib import Path
from urllib.parse import urlsplit

import bcrypt
import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from starlette.middleware.sessions import SessionMiddleware

from shottracker.geometry import Calibration, find_circle
from shottracker.targets import TARGETS, TargetType, deviation_pct, fill_bgr, group_report, score_at, score_points
from shottracker.targets import Hit
from web.analyze import LiveAnalyzer
from web.db import CalibrationRow, HitRow, SessionLocal, ShootingSession, User, init_db, utcnow
from web import storage

logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s %(message)s")
log = logging.getLogger("shottracker.web")
STATIC = Path(__file__).resolve().parent / "static"
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-shottracker-change-me")
MAX_VIDEO = 512 * 1024 * 1024
ANALYZERS: dict[int, LiveAnalyzer] = {}

_DUMMY_HASH = bcrypt.hashpw(b"not-a-real-password", bcrypt.gensalt()).decode("utf-8")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    await asyncio.to_thread(storage.ensure_bucket)
    yield
    for analyzer in list(ANALYZERS.values()):
        analyzer.close()


app = FastAPI(title="Great Shot Target", lifespan=lifespan)
app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    session_cookie="shottracker",
    max_age=14 * 24 * 3600,
    same_site="lax",
    https_only=os.environ.get("COOKIE_SECURE", "") == "1",
)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class AuthIn(BaseModel):
    username: str
    password: str


class SessionPatch(BaseModel):
    note: str | None = None
    carry_note: bool | None = None
    target_key: str | None = None
    video_store: str | None = None
    video_name: str | None = Field(default=None, max_length=200)
    sensitivity: int | None = None
    size_scale: float | None = None


class SettingsIn(BaseModel):
    record: bool | None = None
    server_video: bool | None = None
    zoom: bool | None = None
    hit_label_px: int | None = Field(default=None, ge=10, le=36)


DEFAULT_SETTINGS = {"record": True, "server_video": False, "zoom": True, "hit_label_px": 14}


def user_settings(user: User) -> dict:
    merged = dict(DEFAULT_SETTINGS)
    merged.update({key: value for key, value in (user.settings or {}).items() if key in DEFAULT_SETTINGS})
    return merged


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def check_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        return False


def clean_username(value: str) -> str | None:
    name = value.strip()
    if len(name) < 3 or len(name) > 32 or any(ch.isspace() for ch in name):
        return None
    return name


def current_user(request: Request, db: Session = Depends(get_db)) -> User:
    uid = request.session.get("uid")
    user = db.get(User, uid) if uid else None
    if user is None:
        raise HTTPException(status_code=401, detail="Нужен вход")
    return user


def own_session(db: Session, user: User, session_id: int) -> ShootingSession:
    row = db.get(ShootingSession, session_id)
    if row is None or row.user_id != user.id:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    return row


def page(name: str) -> FileResponse:
    return FileResponse(STATIC / name, headers={"Cache-Control": "no-cache"})


def target_public(target: TargetType) -> dict:
    rings = []
    for ring in target.rings:
        blue, green, red = fill_bgr(target, ring)
        score = ring.score if isinstance(ring.score, str) else int(ring.score)
        rings.append({"score": score, "radius": ring.radius, "color": f"#{red:02x}{green:02x}{blue:02x}"})
    return {"key": target.key, "name": target.name, "kind": target.kind, "rings": rings}


def hits_sorted(row: ShootingSession) -> list[HitRow]:
    return sorted(row.hits, key=lambda hit: hit.idx)


def as_domain(rows: list[HitRow]) -> list[Hit]:
    hits = []
    for row in rows:
        score: int | str = "X" if row.score == "X" else int(row.score)
        hits.append(Hit(row.norm_x, row.norm_y, score, row.created_at, row.source, 0))
    return hits


def hit_public(row: HitRow) -> dict:
    score: int | str = "X" if row.score == "X" else int(row.score)
    return {
        "id": row.id,
        "n": row.idx,
        "norm_x": row.norm_x,
        "norm_y": row.norm_y,
        "score": row.score,
        "deviation": deviation_pct(Hit(row.norm_x, row.norm_y, score, row.created_at, row.source)),
        "source": row.source,
        "offset_ms": row.offset_ms,
        "at": row.created_at.isoformat(),
    }


def session_public(row: ShootingSession) -> dict:
    target = TARGETS.get(row.target_key) or TARGETS["pistol"]
    rows = hits_sorted(row)
    domain = as_domain(rows)
    total = sum(score_points(hit.score, target) for hit in domain)
    count = len(domain)
    devs = [deviation_pct(hit) for hit in domain]
    best = None
    if domain:
        winner = max(domain, key=lambda hit: (score_points(hit.score, target), hit.score == "X"))
        best = "X" if winner.score == "X" else str(winner.score)
    report = group_report(domain)
    group = None
    if report:
        group = {
            "spread": report["spread"],
            "avg_x": report["avg_x"],
            "avg_y": report["avg_y"],
            "direction": report["direction"],
            "centered": report["centered"],
        }
    calib = None
    if row.calibration is not None:
        calib = {
            "center": [row.calibration.center_x, row.calibration.center_y],
            "outer_r": row.calibration.outer_r,
            "frame": [row.calibration.frame_w, row.calibration.frame_h],
        }
    return {
        "id": row.id,
        "number": row.number,
        "status": row.status,
        "target_key": row.target_key,
        "note": row.note,
        "carry_note": row.carry_note,
        "video_store": row.video_store,
        "video_name": row.video_name,
        "has_video": bool(row.video_key),
        "sensitivity": row.sensitivity,
        "size_scale": row.size_scale,
        "started_at": row.started_at.isoformat(),
        "calib": calib,
        "hits": [hit_public(hit) for hit in rows],
        "stats": {
            "count": count,
            "total": total,
            "average": round(total / count, 1) if count else None,
            "best": best,
            "mean_deviation": round(sum(devs) / len(devs), 1) if devs else None,
            "best_deviation": round(min(devs), 1) if devs else None,
        },
        "group": group,
    }


def session_brief(row: ShootingSession, hit_count: int) -> dict:
    return {
        "id": row.id,
        "number": row.number,
        "status": row.status,
        "target_key": row.target_key,
        "started_at": row.started_at.isoformat(),
        "hit_count": hit_count,
        "video_store": row.video_store,
        "video_name": row.video_name,
        "has_video": bool(row.video_key),
    }


def close_open(db: Session, user_id: int, keep_id: int | None = None) -> None:
    rows = db.scalars(
        select(ShootingSession).where(ShootingSession.user_id == user_id, ShootingSession.status == "open")
    ).all()
    now = utcnow()
    for row in rows:
        if row.id == keep_id:
            continue
        row.status = "closed"
        row.ended_at = now


def create_session(db: Session, user: User) -> ShootingSession:
    prev = db.scalar(
        select(ShootingSession)
        .where(ShootingSession.user_id == user.id)
        .order_by(ShootingSession.id.desc())
    )
    close_open(db, user.id)
    note = prev.note if prev and prev.carry_note else ""
    row = ShootingSession(
        user_id=user.id,
        number=(prev.number + 1) if prev else 1,
        target_key=prev.target_key if prev else "pistol",
        note=note,
        carry_note=bool(prev.carry_note) if prev else False,
        sensitivity=prev.sensitivity if prev else 28,
        size_scale=prev.size_scale if prev else 1.0,
    )
    if prev is not None and prev.calibration is not None:
        old = prev.calibration
        row.calibration = CalibrationRow(
            space=old.space,
            center_x=old.center_x,
            center_y=old.center_y,
            outer_r=old.outer_r,
            frame_w=old.frame_w,
            frame_h=old.frame_h,
        )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def calib_from_row(row: CalibrationRow | None) -> Calibration | None:
    if row is None or row.outer_r < 10:
        return None
    return Calibration("raw", (row.center_x, row.center_y), row.outer_r, None, (row.frame_w, row.frame_h))


def save_calib(db: Session, row: ShootingSession, calib: Calibration) -> None:
    width, height = calib.frame_size or (0, 0)
    current = row.calibration
    if current is None:
        current = CalibrationRow(
            session_id=row.id,
            space="raw",
            center_x=calib.center[0],
            center_y=calib.center[1],
            outer_r=calib.outer_r,
            frame_w=width,
            frame_h=height,
        )
        db.add(current)
    else:
        current.center_x = calib.center[0]
        current.center_y = calib.center[1]
        current.outer_r = calib.outer_r
        current.frame_w = width
        current.frame_h = height
    db.commit()
    db.refresh(row)


def rescore(row: ShootingSession) -> None:
    target = TARGETS.get(row.target_key) or TARGETS["pistol"]
    for hit in row.hits:
        hit.score = str(score_at(hit.norm_x, hit.norm_y, target))


def persist_hits(session_id: int, raw_hits: list[dict]) -> list[dict]:
    if not raw_hits:
        return []
    with SessionLocal() as db:
        row = db.get(ShootingSession, session_id)
        if row is None or row.status != "open":
            return []
        target = TARGETS.get(row.target_key) or TARGETS["pistol"]
        current = db.scalar(select(func.count()).select_from(HitRow).where(HitRow.session_id == session_id)) or 0
        saved: list[HitRow] = []
        for item in raw_hits:
            if current >= 500:
                break
            current += 1
            source = item.get("source") if item.get("source") in {"camera", "manual"} else "camera"
            norm_x = float(item["norm_x"])
            norm_y = float(item["norm_y"])
            hit = HitRow(
                session_id=session_id,
                idx=int(current),
                norm_x=norm_x,
                norm_y=norm_y,
                score=str(score_at(norm_x, norm_y, target)),
                source=source,
                offset_ms=max(0, int(item.get("offset_ms") or 0)),
            )
            db.add(hit)
            saved.append(hit)
        db.commit()
        for hit in saved:
            db.refresh(hit)
        return [hit_public(hit) for hit in saved]


def load_state(session_id: int) -> dict | None:
    with SessionLocal() as db:
        row = db.get(ShootingSession, session_id)
        if row is None:
            return None
        return session_public(row)


def allowed_camera_url(url: str) -> bool:
    parts = urlsplit(url.strip())
    return parts.scheme in {"rtsp", "rtsps", "rtmp", "http", "https"} and bool(parts.netloc)


def push_analyzer(row: ShootingSession) -> None:
    analyzer = ANALYZERS.get(row.id)
    if analyzer is None:
        return
    analyzer.set_target(row.target_key)
    analyzer.set_sensitivity(row.sensitivity)
    analyzer.set_size(row.size_scale)


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse("/app")


@app.get("/login")
def login_page() -> FileResponse:
    return page("login.html")


@app.get("/register")
def register_page() -> FileResponse:
    return page("register.html")


@app.get("/app")
def app_page() -> FileResponse:
    return page("app.html")


@app.get("/api/health")
def health() -> dict:
    with SessionLocal() as db:
        db.execute(select(1))
    return {"ok": True}


@app.get("/api/targets")
def targets() -> list[dict]:
    return [target_public(TARGETS[key]) for key in TARGETS]


@app.post("/api/register")
def register(body: AuthIn, request: Request, db: Session = Depends(get_db)) -> dict:
    username = clean_username(body.username)
    if username is None:
        raise HTTPException(status_code=400, detail="Логин — от 3 до 32 знаков, без пробелов")
    if not 8 <= len(body.password) <= 72:
        raise HTTPException(status_code=400, detail="Пароль — от 8 до 72 знаков")
    user = User(username=username, password_hash=hash_password(body.password))
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Такой логин уже занят") from None
    db.refresh(user)
    request.session["uid"] = user.id
    return {"id": user.id, "username": user.username}


@app.post("/api/login")
def login(body: AuthIn, request: Request, db: Session = Depends(get_db)) -> dict:
    username = clean_username(body.username) or ""
    user = db.scalar(select(User).where(User.username == username)) if username else None
    hashed = user.password_hash if user else _DUMMY_HASH
    if user is None or not check_password(body.password, hashed):
        raise HTTPException(status_code=401, detail="Неверный логин или пароль")
    request.session["uid"] = user.id
    return {"id": user.id, "username": user.username}


@app.post("/api/logout")
def logout(request: Request) -> dict:
    request.session.clear()
    return {"ok": True}


@app.get("/api/me")
def me(user: User = Depends(current_user)) -> dict:
    return {"id": user.id, "username": user.username, "settings": user_settings(user)}


@app.put("/api/settings")
def save_settings(body: SettingsIn, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    merged = user_settings(user)
    merged.update(body.model_dump(exclude_none=True))
    user.settings = merged
    db.commit()
    return merged


@app.get("/api/sessions")
def list_sessions(user: User = Depends(current_user), db: Session = Depends(get_db)) -> list[dict]:
    rows = db.scalars(
        select(ShootingSession).where(ShootingSession.user_id == user.id).order_by(ShootingSession.id.desc())
    ).all()
    if not rows:
        return []
    ids = [row.id for row in rows]
    counts = dict(
        db.execute(
            select(HitRow.session_id, func.count()).where(HitRow.session_id.in_(ids)).group_by(HitRow.session_id)
        ).all()
    )
    return [session_brief(row, int(counts.get(row.id, 0))) for row in rows]


@app.get("/api/sessions/current")
def current_session(user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    row = db.scalar(
        select(ShootingSession)
        .where(ShootingSession.user_id == user.id, ShootingSession.status == "open")
        .order_by(ShootingSession.id.desc())
    )
    if row is None:
        row = create_session(db, user)
    else:
        close_open(db, user.id, keep_id=row.id)
        db.commit()
        db.refresh(row)
    return session_public(row)


@app.post("/api/sessions")
def new_session(user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    return session_public(create_session(db, user))


@app.get("/api/sessions/{session_id}")
def read_session(session_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    return session_public(own_session(db, user, session_id))


@app.patch("/api/sessions/{session_id}")
def patch_session(
    session_id: int,
    body: SessionPatch,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> dict:
    row = own_session(db, user, session_id)
    if body.note is not None:
        row.note = body.note.strip()[:2000]
    if body.carry_note is not None:
        row.carry_note = body.carry_note
    if body.video_store is not None:
        if body.video_store not in {"local", "server"}:
            raise HTTPException(status_code=400, detail="Хранилище видео: local или server")
        row.video_store = body.video_store
    if body.video_name is not None:
        row.video_name = body.video_name.strip() or None
    if body.sensitivity is not None:
        row.sensitivity = int(min(80, max(8, body.sensitivity)))
    if body.size_scale is not None:
        row.size_scale = float(min(2.5, max(0.4, body.size_scale)))
    if body.target_key is not None:
        if body.target_key not in TARGETS:
            raise HTTPException(status_code=400, detail="Неизвестная мишень")
        row.target_key = body.target_key
        rescore(row)
    db.commit()
    db.refresh(row)
    push_analyzer(row)
    return session_public(row)


@app.post("/api/sessions/{session_id}/note/copy")
def copy_note(session_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    row = own_session(db, user, session_id)
    prev = db.scalar(
        select(ShootingSession)
        .where(ShootingSession.user_id == user.id, ShootingSession.id < row.id)
        .order_by(ShootingSession.id.desc())
    )
    if prev is None or not prev.note.strip():
        raise HTTPException(status_code=400, detail="Прошлая заметка пуста")
    row.note = prev.note
    db.commit()
    db.refresh(row)
    return session_public(row)


@app.post("/api/sessions/{session_id}/undo")
def undo_hit(session_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    row = own_session(db, user, session_id)
    if row.status != "open":
        raise HTTPException(status_code=400, detail="Сессия уже закрыта")
    last = db.scalar(
        select(HitRow).where(HitRow.session_id == row.id).order_by(HitRow.idx.desc())
    )
    if last is None:
        raise HTTPException(status_code=400, detail="Нечего отменять")
    analyzer = ANALYZERS.get(row.id)
    if analyzer is not None:
        analyzer.undo_point(last.norm_x, last.norm_y)
    db.delete(last)
    db.commit()
    db.refresh(row)
    return session_public(row)


@app.post("/api/sessions/{session_id}/auto-calib")
async def auto_calib(
    session_id: int,
    file: UploadFile = File(...),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> dict:
    row = own_session(db, user, session_id)
    if row.status != "open":
        raise HTTPException(status_code=400, detail="Сессия уже закрыта")
    data = await file.read()
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Кадр не прочитался")
    analyzer = ANALYZERS.get(row.id)
    if analyzer is not None:
        calib = analyzer.auto_from_frame(frame)
    else:
        found = find_circle(frame)
        height, width = frame.shape[:2]
        calib = None if found is None else Calibration("raw", (found[0], found[1]), found[2], None, (width, height))
    if calib is None:
        raise HTTPException(status_code=400, detail="Круг на кадре не нашёлся")
    save_calib(db, row, calib)
    db.refresh(row)
    return session_public(row)


@app.post("/api/sessions/{session_id}/video")
async def upload_video(
    session_id: int,
    file: UploadFile = File(...),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> dict:
    row = own_session(db, user, session_id)
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        size += len(chunk)
        if size > MAX_VIDEO:
            raise HTTPException(status_code=413, detail="Видео больше 512 МБ")
        chunks.append(chunk)
    if size == 0:
        raise HTTPException(status_code=400, detail="Пустой файл")
    name = (file.filename or f"session_{row.number}.webm").replace("/", "_")[-180:]
    key = f"users/{user.id}/sessions/{row.id}.webm"
    try:
        await asyncio.to_thread(storage.put_bytes, key, b"".join(chunks), file.content_type or "video/webm")
    except Exception as exc:
        log.warning("upload failed: %s", exc)
        raise HTTPException(status_code=502, detail="Не удалось положить видео в хранилище") from None
    row.video_key = key
    row.video_name = name
    row.video_store = "server"
    db.commit()
    db.refresh(row)
    return session_public(row)


@app.get("/api/sessions/{session_id}/video")
def download_video(session_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    row = own_session(db, user, session_id)
    if not row.video_key:
        raise HTTPException(status_code=404, detail="На сервере этого видео нет")
    try:
        obj = storage.get_object(row.video_key)
    except Exception as exc:
        log.warning("download failed: %s", exc)
        raise HTTPException(status_code=502, detail="Хранилище не отдало видео") from None
    media = obj.get("ContentType") or "video/webm"

    def chunks():
        body = obj["Body"]
        try:
            while True:
                data = body.read(1024 * 1024)
                if not data:
                    break
                yield data
        finally:
            body.close()

    return StreamingResponse(chunks(), media_type=media)


async def emit(ws: WebSocket, session_id: int, result: dict, lock: asyncio.Lock) -> None:
    saved = await asyncio.to_thread(persist_hits, session_id, result.get("hits") or [])
    status = result.get("status") or ""
    analyzer = ANALYZERS.get(session_id)
    if saved:
        state = await asyncio.to_thread(load_state, session_id)
        async with lock:
            await ws.send_json({"type": "hits", "hits": saved, "state": state})
        if analyzer is not None:
            analyzer.last_status = "ok"
        return
    if analyzer is not None and status == analyzer.last_status:
        return
    if analyzer is not None:
        analyzer.last_status = status
    if status:
        async with lock:
            await ws.send_json({"type": "status", "status": status})


async def pump(ws: WebSocket, analyzer: LiveAnalyzer, session_id: int, lock: asyncio.Lock) -> None:
    while not analyzer.stopped and analyzer.mode in {"demo", "ip"}:
        packet = await asyncio.to_thread(analyzer.grab)
        if analyzer.stopped or analyzer.mode not in {"demo", "ip"}:
            break
        if packet is None:
            await asyncio.sleep(0.3)
            continue
        jpeg, result = packet
        try:
            async with lock:
                await ws.send_bytes(jpeg)
            await emit(ws, session_id, result, lock)
        except Exception:
            break
        await asyncio.sleep(0.08)


class FrameStats:
    def __init__(self, every: float = 30.0) -> None:
        self.every = every
        self.since = time.monotonic()
        self.frames = 0
        self.total_ms = 0.0
        self.max_ms = 0.0
        self.statuses: Counter[str] = Counter()

    def add(self, status: str, ms: float) -> None:
        self.frames += 1
        self.total_ms += ms
        self.max_ms = max(self.max_ms, ms)
        self.statuses[status] += 1

    def due(self) -> bool:
        return time.monotonic() - self.since >= self.every

    def flush(self) -> str:
        seconds = max(0.001, time.monotonic() - self.since)
        mean = self.total_ms / self.frames if self.frames else 0.0
        line = (
            f"{self.frames / seconds:.1f} fps, {mean:.0f} ms avg, {self.max_ms:.0f} ms max, "
            f"{dict(self.statuses.most_common())}"
        )
        self.__init__(self.every)
        return line


def ws_user_id(ws: WebSocket) -> int | None:
    try:
        uid = ws.session.get("uid")
    except AssertionError:
        return None
    return int(uid) if uid else None


@app.websocket("/api/live/{session_id}")
async def live(ws: WebSocket, session_id: int) -> None:
    uid = ws_user_id(ws)
    await ws.accept()
    if not uid:
        await ws.close(code=4401)
        return
    with SessionLocal() as db:
        user = db.get(User, uid)
        row = db.get(ShootingSession, session_id)
        if user is None or row is None or row.user_id != user.id:
            await ws.send_json({"type": "error", "message": "Сессия не найдена"})
            await ws.close()
            return
        points = [(hit.norm_x, hit.norm_y) for hit in hits_sorted(row)]
        calib = calib_from_row(row.calibration)
        target_key = row.target_key
        sensitivity = row.sensitivity
        size_scale = row.size_scale
        open_session = row.status == "open"
    old = ANALYZERS.get(session_id)
    if old is not None:
        old.close()
    analyzer = LiveAnalyzer()
    analyzer.configure(target_key, sensitivity, size_scale, points, calib)
    ANALYZERS[session_id] = analyzer
    pump_task: asyncio.Task | None = None
    lock = asyncio.Lock()
    stats = FrameStats()

    async def send(payload: dict) -> None:
        async with lock:
            await ws.send_json(payload)

    try:
        state = await asyncio.to_thread(load_state, session_id)
        await send({"type": "state", "state": state, "mode": analyzer.mode})
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                if analyzer.mode != "browser" or not open_session:
                    await send({"type": "tick", "status": "paused" if open_session else "closed", "ms": 0})
                    continue
                started = time.perf_counter()
                result = await asyncio.to_thread(analyzer.step_jpeg, message["bytes"])
                spent = (time.perf_counter() - started) * 1000
                await emit(ws, session_id, result, lock)
                status = result.get("status") or ""
                await send({"type": "tick", "status": status, "ms": round(spent)})
                stats.add(status, spent)
                if stats.due():
                    log.info("live session %s: %s", session_id, stats.flush())
                continue
            text = message.get("text")
            if not text:
                continue
            try:
                command = json.loads(text)
            except json.JSONDecodeError:
                continue
            kind = command.get("type")
            if kind == "demo":
                if not open_session:
                    await send({"type": "error", "message": "Сессия закрыта. Нажмите «Новая»"})
                    continue
                analyzer.start_demo()
                demo_calib = analyzer.calib
                await send({
                    "type": "mode",
                    "mode": "demo",
                    "calib": None if demo_calib is None else {
                        "center": list(demo_calib.center),
                        "outer_r": demo_calib.outer_r,
                        "frame": list(demo_calib.frame_size or (0, 0)),
                    },
                })
                if pump_task is None or pump_task.done():
                    pump_task = asyncio.create_task(pump(ws, analyzer, session_id, lock))
            elif kind == "browser":
                saved = calib
                with SessionLocal() as db:
                    fresh = db.get(ShootingSession, session_id)
                    saved = calib_from_row(fresh.calibration) if fresh else None
                analyzer.start_browser(saved)
                await send({"type": "mode", "mode": "browser"})
            elif kind == "ip":
                if not open_session:
                    await send({"type": "error", "message": "Сессия закрыта. Нажмите «Новая»"})
                    continue
                url = str(command.get("url") or "").strip()
                if not allowed_camera_url(url):
                    await send({"type": "error", "message": "Нужен адрес rtsp://, http:// или https://"})
                    continue
                with SessionLocal() as db:
                    fresh = db.get(ShootingSession, session_id)
                    saved_calib = calib_from_row(fresh.calibration) if fresh else None
                error = await asyncio.to_thread(analyzer.start_ip, url, saved_calib)
                if error:
                    await send({"type": "error", "message": error})
                    continue
                await send({"type": "mode", "mode": "ip"})
                if pump_task is None or pump_task.done():
                    pump_task = asyncio.create_task(pump(ws, analyzer, session_id, lock))
            elif kind == "calib":
                if not open_session:
                    await send({"type": "error", "message": "Сессия закрыта"})
                    continue
                if analyzer.mode == "demo":
                    await send({"type": "error", "message": "В демо мишень уже размечена"})
                    continue
                try:
                    cx, cy = float(command["cx"]), float(command["cy"])
                    ex, ey = float(command["ex"]), float(command["ey"])
                    width, height = int(command["w"]), int(command["h"])
                except (KeyError, TypeError, ValueError):
                    await send({"type": "error", "message": "Калибровка не разобралась"})
                    continue
                radius = hypot(ex - cx, ey - cy)
                if radius < 10 or width < 2 or height < 2:
                    await send({"type": "error", "message": "Край зоны 1 слишком близко к центру"})
                    continue
                fresh = Calibration("raw", (cx, cy), radius, None, (width, height))
                analyzer.set_calib(fresh)
                with SessionLocal() as db:
                    row = db.get(ShootingSession, session_id)
                    if row is not None:
                        save_calib(db, row, fresh)
                state = await asyncio.to_thread(load_state, session_id)
                await send({"type": "state", "state": state, "mode": analyzer.mode})
            elif kind == "start":
                if not open_session:
                    await send({"type": "error", "message": "Сессия закрыта. Нажмите «Новая»"})
                    continue
                error = analyzer.start()
                if error:
                    await send({"type": "error", "message": error})
                else:
                    await send({"type": "started"})
            elif kind == "stop":
                analyzer.stop()
                await send({"type": "stopped"})
            elif kind == "reference":
                analyzer.reset_reference()
                analyzer.last_status = ""
                await send({"type": "status", "status": "warmup"})
            elif kind == "manual":
                if not open_session:
                    await send({"type": "error", "message": "Сессия закрыта"})
                    continue
                try:
                    packed = analyzer.manual(float(command["norm_x"]), float(command["norm_y"]))
                except (KeyError, TypeError, ValueError):
                    packed = None
                if packed is None:
                    await send({"type": "error", "message": "Точка слишком далеко от мишени или уже есть"})
                else:
                    await emit(ws, session_id, {"status": "ok", "hits": [packed]}, lock)
    except WebSocketDisconnect:
        pass
    finally:
        analyzer.close()
        if pump_task is not None:
            pump_task.cancel()
        if ANALYZERS.get(session_id) is analyzer:
            ANALYZERS.pop(session_id, None)
