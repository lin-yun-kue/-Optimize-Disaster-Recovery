from __future__ import annotations

from typing import Final

# Canonical disaster key names used by calendar/weather, priors, and profiles.
DISASTER_KEYS: Final[tuple[str, ...]] = ("landslide", "snow", "wildfire_debris", "flood")

# Class-name -> canonical key.
CLASS_NAME_TO_KEY: Final[dict[str, str]] = {
    "Landslide": "landslide",
    "SnowEvent": "snow",
    "WildfireDebris": "wildfire_debris",
    "FloodEvent": "flood",
}

# Canonical key -> class-name helper.
KEY_TO_CLASS_NAME: Final[dict[str, str]] = {v: k for k, v in CLASS_NAME_TO_KEY.items()}


def disaster_key_for_instance_name(class_name: str) -> str | None:
    return CLASS_NAME_TO_KEY.get(class_name)
