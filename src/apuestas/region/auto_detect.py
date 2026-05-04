"""Auto-detección región IP → ajusta flags books MX/US dinámicamente.

Flujo:
1. Query ipinfo.io/country (free, sin auth) con timeout 3s
2. Si country == "US" → activa DK/FD/MGM + US_VPN_ACTIVE=true
3. Si country != "US" (típicamente MX) → desactiva DK/FD/MGM + US_VPN_ACTIVE=false
4. Los books MX (Caliente/Codere/Winpot) + offshore (BetUS/etc.) siempre activos

Se invoca desde:
- CLI `apuestas` (al arrancar)
- Bot Telegram (en startup)
- systemd timers (al disparo)

Persiste a .env para que próximos runs tengan la config correcta.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Literal

import httpx

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

Region = Literal["US", "MX", "OTHER"]

# Flags que se toggle por región
US_REQUIRED_FLAGS = {
    "APUESTAS_US_VPN_ACTIVE": ("true", "false"),
    "APUESTAS_ENABLE_DK": ("true", "false"),
    "APUESTAS_ENABLE_FANDUEL": ("true", "false"),
    "APUESTAS_ENABLE_BETMGM": ("true", "false"),
}


async def detect_country(*, timeout: float = 3.0) -> str | None:
    """Query múltiples APIs geo-IP con fallback. Returns 2-letter country code.

    Fuentes en orden (todas free, sin auth):
    1. cloudflare trace (cf-ipcountry) — ultra reliable, sin rate-limit
    2. ifconfig.co/country-iso
    3. api.country.is (JSON)
    4. ipapi.co/country
    5. ipinfo.io/country (puede tener rate-limit)
    """
    sources: list[tuple[str, str]] = [
        ("cloudflare", "https://www.cloudflare.com/cdn-cgi/trace"),
        ("ifconfig_co", "https://ifconfig.co/country-iso"),
        ("country_is", "https://api.country.is/"),
        ("ipapi_co", "https://ipapi.co/country/"),
        ("ipinfo", "https://ipinfo.io/country"),
    ]

    async def _query_one(name: str, url: str) -> str | None:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(url, headers={"User-Agent": "apuestas-bot/1.0"})
            if resp.status_code != 200:
                logger.info("region.source_non200", source=name, status=resp.status_code)
                return None
            text = resp.text.strip()
            if name == "cloudflare":
                for line in text.splitlines():
                    if line.startswith("loc="):
                        return line.split("=", 1)[1].strip().upper()[:2] or None
                return None
            if name == "country_is":
                try:
                    import json as _json

                    data = _json.loads(text)
                    return (data.get("country") or "").upper()[:2] or None
                except Exception:
                    return None
            code = text.upper()[:2]
            if code in ("{ ", "{", "IP", "ER"):
                return None
            return code if len(code) == 2 and code.isalpha() else None
        except (TimeoutError, httpx.HTTPError) as exc:
            logger.info("region.source_fail", source=name, error=str(exc)[:80])
            return None

    for name, url in sources:
        code = await _query_one(name, url)
        if code:
            logger.info("region.detected", source=name, country=code)
            return code
    logger.warning("region.all_sources_failed")
    return None


def classify_region(country: str | None) -> Region:
    if country == "US":
        return "US"
    if country == "MX":
        return "MX"
    return "OTHER"


def _parse_env_lines(env_path: Path) -> list[str]:
    if not env_path.exists():
        return []
    return env_path.read_text(encoding="utf-8").splitlines()


def _write_env_lines(env_path: Path, lines: list[str]) -> None:
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def apply_region_flags(region: Region, *, env_path: Path = Path(".env")) -> dict[str, str]:
    """Actualiza .env con flags apropiados a la región. Idempotente."""
    values: dict[str, str] = {}
    for key, (us_val, non_us_val) in US_REQUIRED_FLAGS.items():
        values[key] = us_val if region == "US" else non_us_val

    lines = _parse_env_lines(env_path)
    seen: set[str] = set()
    new_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            new_lines.append(line)
            continue
        key, _, _v = stripped.partition("=")
        key = key.strip()
        if key in values:
            new_lines.append(f"{key}={values[key]}")
            seen.add(key)
        else:
            new_lines.append(line)
    for key, val in values.items():
        if key not in seen:
            new_lines.append(f"{key}={val}")

    _write_env_lines(env_path, new_lines)

    # Aplicar al environment actual del proceso también
    for key, val in values.items():
        os.environ[key] = val

    return values


async def auto_configure_region(*, env_path: Path = Path(".env")) -> dict[str, object]:
    """Orchestration: detect → classify → apply → log. Idempotente."""
    country = await detect_country()
    region = classify_region(country)
    flags = apply_region_flags(region, env_path=env_path)
    logger.info(
        "region.auto_configured",
        country=country or "unknown",
        region=region,
        vpn_active=flags.get("APUESTAS_US_VPN_ACTIVE") == "true",
        us_books_enabled=flags.get("APUESTAS_ENABLE_DK") == "true",
    )
    return {
        "country": country,
        "region": region,
        "flags_applied": flags,
    }


if __name__ == "__main__":
    import json

    result = asyncio.run(auto_configure_region())
    print(json.dumps(result, indent=2))
