"""Región / geo detection utilities."""

from apuestas.region.auto_detect import (
    apply_region_flags,
    auto_configure_region,
    classify_region,
    detect_country,
)

__all__ = [
    "apply_region_flags",
    "auto_configure_region",
    "classify_region",
    "detect_country",
]
