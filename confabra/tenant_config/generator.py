"""
Tenant config generator — produces per-profile ``tenant_config.json`` artifacts.

Each profile gets exactly one ``tenant_config.json`` written to ``output_dir``.
The file is a serialized :class:`~confabra.schemas.TenantConfig` and is
accompanied by a SHA-256 content hash returned to the caller for embedding in
the corpus manifest.

Public API
----------
- :func:`generate_tenant_config` — build, write, and hash the config
- :func:`load_tenant_config` — deserialise a previously-written config file
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from confabra.profiles.base import Profile
from confabra.schemas import BrandVoiceVariant, CoachingStyleOverlay, TenantConfig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _variant_to_dict(variant: BrandVoiceVariant) -> dict[str, Any]:
    """
    Serialise a :class:`BrandVoiceVariant` to a plain JSON-compatible dict.

    ``feature_profile`` is intentionally included so that the config file is
    self-contained — the prose generator and validator can reconstruct the full
    variant without re-importing the profile module.
    """
    return {
        "id": variant.id,
        "label": variant.label,
        "content": variant.content,
        "feature_profile": variant.feature_profile,
    }


def _overlay_to_dict(overlay: CoachingStyleOverlay) -> dict[str, Any]:
    """Serialise a :class:`CoachingStyleOverlay` to a plain JSON-compatible dict."""
    return {
        "id": overlay.id,
        "label": overlay.label,
        "content": overlay.content,
        "brand_voice_variant_id": overlay.brand_voice_variant_id,
    }


def _build_config_dict(profile: Profile) -> dict[str, Any]:
    """
    Build the raw config dict for *profile*.

    Structure matches the :class:`~confabra.schemas.TenantConfig` schema:
    - ``tenant_id`` — derived from profile name and version (stable across reruns)
    - ``brand_voice_variants`` — all variants returned by the profile
    - ``coaching_style_overlays`` — always ``[]`` at scaffolding stage (P1-7)
    - ``knowledge_base_subset`` — ``"all"`` (no subsetting in v1)
    - ``rubric_weights`` — platform defaults (empathy 0.25, resolution 0.30,
      brand_voice 0.20, accuracy 0.25); sum is 1.0

    The profile is responsible for returning the correct variant count
    (SaaS: 3, PS: 1).  This function validates only that the list is
    non-empty; a zero-variant result is always a profile implementation bug.
    """
    tenant_id = f"tenant_{profile.name}_v1"

    variants = profile.brand_voice_variants()
    if not variants:
        raise ValueError(
            f"Profile '{profile.name}' returned no brand voice variants"
        )
    brand_voice_dicts = [_variant_to_dict(v) for v in variants]

    # coaching_style_overlays is always empty at the scaffolding stage;
    # P1-7 will populate this field in a later task.
    overlays: list[dict[str, Any]] = []

    rubric_weights: dict[str, float] = {
        "empathy": 0.25,
        "resolution": 0.30,
        "brand_voice": 0.20,
        "accuracy": 0.25,
    }

    return {
        "tenant_id": tenant_id,
        "brand_voice_variants": brand_voice_dicts,
        "coaching_style_overlays": overlays,
        "knowledge_base_subset": "all",
        "rubric_weights": rubric_weights,
    }


def _sha256_of_dict(data: dict[str, Any]) -> str:
    """
    Compute a stable SHA-256 hex digest of *data*.

    Keys are sorted before serialisation to guarantee that logically identical
    configs produce the same hash regardless of dict insertion order.
    """
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_tenant_config(
    profile: Profile,
    output_dir: Path,
) -> tuple[dict[str, Any], str]:
    """
    Generate and persist the tenant config for *profile*.

    Steps:
    1. Build the config dict from the profile's brand voice variants and
       platform-default rubric weights.
    2. Serialise to canonical JSON (keys sorted, UTF-8 encoded).
    3. Compute SHA-256 of the canonical JSON.
    4. Write ``tenant_config.json`` to *output_dir* (created if absent).
    5. Return ``(config_dict, content_hash)``.

    The returned ``content_hash`` is a SHA-256 hex digest suitable for embedding
    in the corpus manifest's ``tenant_config_hash`` field.

    The profile is responsible for returning the correct variant count
    (SaaS: 3, PS: 1).  The generator trusts the profile but raises
    :exc:`ValueError` if the profile returns zero variants.

    Parameters
    ----------
    profile:
        A concrete :class:`~confabra.profiles.base.Profile` instance (SaaS or PS).
    output_dir:
        Directory to write ``tenant_config.json`` into.  Created if it does not
        exist.

    Returns
    -------
    tuple[dict, str]
        ``(config_dict, sha256_hex)`` — the serialisable config and its hash.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_dict = _build_config_dict(profile)
    content_hash = _sha256_of_dict(config_dict)

    dest = output_dir / "tenant_config.json"
    canonical_json = json.dumps(config_dict, sort_keys=True, indent=2, ensure_ascii=False)
    dest.write_text(canonical_json, encoding="utf-8")

    return config_dict, content_hash


def load_tenant_config(path: Path) -> dict[str, Any]:
    """
    Load a previously-written ``tenant_config.json`` from *path*.

    The returned dict matches the shape produced by :func:`generate_tenant_config`
    and is compatible with :class:`~confabra.schemas.TenantConfig` for Pydantic
    validation if needed.

    Parameters
    ----------
    path:
        Absolute path to a ``tenant_config.json`` file.

    Returns
    -------
    dict
        Parsed JSON content of the config file.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    json.JSONDecodeError
        If the file content is not valid JSON.
    """
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))
