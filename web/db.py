"""Пользователи, сессии и попадания в Postgres."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, create_engine, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://shottracker:shottracker@localhost:5432/shottracker",
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(200))
    settings: Mapped[dict] = mapped_column(JSONB, default=dict, server_default=text("'{}'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    sessions: Mapped[list[ShootingSession]] = relationship(back_populates="user")


class ShootingSession(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    number: Mapped[int] = mapped_column(Integer)
    target_key: Mapped[str] = mapped_column(String(32), default="pistol")
    note: Mapped[str] = mapped_column(Text, default="")
    carry_note: Mapped[bool] = mapped_column(default=False)
    video_store: Mapped[str] = mapped_column(String(16), default="local")
    video_key: Mapped[str | None] = mapped_column(String(300), nullable=True)
    video_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="open")
    sensitivity: Mapped[int] = mapped_column(Integer, default=28)
    size_scale: Mapped[float] = mapped_column(Float, default=1.0)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    user: Mapped[User] = relationship(back_populates="sessions")
    hits: Mapped[list[HitRow]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    calibration: Mapped[CalibrationRow | None] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        uselist=False,
        lazy="selectin",
    )


class HitRow(Base):
    __tablename__ = "hits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), index=True)
    idx: Mapped[int] = mapped_column(Integer)
    norm_x: Mapped[float] = mapped_column(Float)
    norm_y: Mapped[float] = mapped_column(Float)
    score: Mapped[str] = mapped_column(String(8))
    source: Mapped[str] = mapped_column(String(16))
    offset_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    session: Mapped[ShootingSession] = relationship(back_populates="hits")


class CalibrationRow(Base):
    __tablename__ = "calibrations"

    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), primary_key=True)
    space: Mapped[str] = mapped_column(String(16), default="raw")
    center_x: Mapped[float] = mapped_column(Float)
    center_y: Mapped[float] = mapped_column(Float)
    outer_r: Mapped[float] = mapped_column(Float)
    frame_w: Mapped[int] = mapped_column(Integer)
    frame_h: Mapped[int] = mapped_column(Integer)
    session: Mapped[ShootingSession] = relationship(back_populates="calibration")


engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db() -> None:
    last: Exception | None = None
    for _ in range(30):
        try:
            with engine.connect() as conn:
                conn.execute(text("select 1"))
            Base.metadata.create_all(engine)
            with engine.begin() as conn:
                conn.execute(text(
                    "alter table users add column if not exists settings jsonb not null default '{}'::jsonb"
                ))
            return
        except Exception as exc:
            last = exc
            time.sleep(1)
    raise RuntimeError(f"Postgres недоступен: {last}")
