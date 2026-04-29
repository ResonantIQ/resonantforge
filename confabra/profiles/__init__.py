"""Profile registry for Layer 2 implementations."""
from __future__ import annotations
from confabra.profiles.saas import SaaSProfile
from confabra.profiles.ps import PSProfile
from confabra.profiles.base import Profile  # re-export for type annotations

PROFILES: dict[str, Profile] = {
    "saas": SaaSProfile(),
    "ps": PSProfile(),
    "professional_services": PSProfile(),
}


def get_profile(name: str) -> Profile:
    """
    Look up a registered profile by name.

    Raises ValueError for unknown profile names so callers get a clear message
    rather than a silent None.  The 'professional_services' alias maps to PSProfile.
    """
    profile = PROFILES.get(name)
    if profile is None:
        raise ValueError(f"Unknown profile: {name!r}. Available: {list(PROFILES.keys())}")
    return profile
