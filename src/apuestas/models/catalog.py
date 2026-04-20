"""Modelos ORM de catálogos: sports, leagues, teams, players, venues."""

from __future__ import annotations

from datetime import date
from typing import Any

from sqlalchemy import BigInteger, Boolean, ForeignKey, Integer, Numeric, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from apuestas.db import Base


class Sport(Base):
    __tablename__ = "sports"

    code: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    has_draws: Mapped[bool] = mapped_column(Boolean, default=False, init=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, init=False)


class League(Base):
    __tablename__ = "leagues"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, init=False)
    sport_code: Mapped[str] = mapped_column(Text, ForeignKey("sports.code"))
    name: Mapped[str] = mapped_column(Text, nullable=False)
    external_id: Mapped[str | None] = mapped_column(Text, unique=True, default=None)
    country: Mapped[str | None] = mapped_column(Text, default=None)
    tier: Mapped[int | None] = mapped_column(Integer, default=None)
    meta: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB, default=None)


class Venue(Base):
    __tablename__ = "venues"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, init=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    external_id: Mapped[str | None] = mapped_column(Text, unique=True, default=None)
    city: Mapped[str | None] = mapped_column(Text, default=None)
    country: Mapped[str | None] = mapped_column(Text, default=None)
    timezone: Mapped[str | None] = mapped_column(Text, default=None)
    lat: Mapped[float | None] = mapped_column(Numeric(9, 6), default=None)
    lon: Mapped[float | None] = mapped_column(Numeric(9, 6), default=None)
    altitude_m: Mapped[int | None] = mapped_column(Integer, default=None)
    capacity: Mapped[int | None] = mapped_column(Integer, default=None)
    surface: Mapped[str | None] = mapped_column(Text, default=None)
    roof: Mapped[str | None] = mapped_column(Text, default=None)
    meta: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB, default=None)


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, init=False)
    sport_code: Mapped[str] = mapped_column(Text, ForeignKey("sports.code"))
    name: Mapped[str] = mapped_column(Text, nullable=False)
    external_id: Mapped[str | None] = mapped_column(Text, unique=True, default=None)
    league_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("leagues.id"), default=None
    )
    short_name: Mapped[str | None] = mapped_column(Text, default=None)
    city: Mapped[str | None] = mapped_column(Text, default=None)
    country: Mapped[str | None] = mapped_column(Text, default=None)
    abbreviation: Mapped[str | None] = mapped_column(Text, default=None)
    venue_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("venues.id"), default=None)
    active: Mapped[bool] = mapped_column(Boolean, default=True, init=False)
    meta: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB, default=None)


class Player(Base):
    __tablename__ = "players"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, init=False)
    sport_code: Mapped[str] = mapped_column(Text, ForeignKey("sports.code"))
    full_name: Mapped[str] = mapped_column(Text, nullable=False)
    external_id: Mapped[str | None] = mapped_column(Text, unique=True, default=None)
    team_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("teams.id"), default=None)
    first_name: Mapped[str | None] = mapped_column(Text, default=None)
    last_name: Mapped[str | None] = mapped_column(Text, default=None)
    position: Mapped[str | None] = mapped_column(Text, default=None)
    jersey_number: Mapped[int | None] = mapped_column(Integer, default=None)
    birthdate: Mapped[date | None] = mapped_column(default=None)
    height_cm: Mapped[int | None] = mapped_column(Integer, default=None)
    weight_kg: Mapped[float | None] = mapped_column(Numeric(6, 2), default=None)
    active: Mapped[bool] = mapped_column(Boolean, default=True, init=False)
    meta: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB, default=None)
