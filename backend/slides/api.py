"""
Low-level Google Slides API helpers: constants, URL parsing, fetching, batch updates.
"""

import re
import uuid
from typing import Optional

import httpx

SLIDES_API = "https://slides.googleapis.com/v1/presentations"

DEFAULT_PAGE_WIDTH_EMU = 9_144_000
DEFAULT_PAGE_HEIGHT_EMU = 6_858_000
PT_TO_EMU = 12_700


def parse_slides_url(url: str) -> Optional[tuple[str, Optional[str]]]:
    """Extract (presentation_id, page_object_id) from a Google Slides URL."""
    if not url or "docs.google.com/presentation" not in url:
        return None
    pres_match = re.search(r"/presentation/d/([a-zA-Z0-9_-]+)", url)
    if not pres_match:
        return None
    presentation_id = pres_match.group(1)
    page_match = re.search(r"#slide=id\.([a-zA-Z0-9_p.]+)", url)
    page_id = page_match.group(1) if page_match else None
    return (presentation_id, page_id)


def get_full_presentation(presentation_id: str, access_token: str) -> dict:
    """Fetch the entire presentation (all slides, elements, page size) in one API call."""
    headers = {"Authorization": f"Bearer {access_token}"}
    r = httpx.get(
        f"{SLIDES_API}/{presentation_id}",
        headers=headers,
        timeout=30.0,
    )
    r.raise_for_status()
    return r.json()


def get_slide_elements(
    presentation_id: str, page_id: str, access_token: str
) -> dict:
    """Fetch a single slide's page elements via presentations.pages.get."""
    headers = {"Authorization": f"Bearer {access_token}"}
    r = httpx.get(
        f"{SLIDES_API}/{presentation_id}/pages/{page_id}",
        headers=headers,
        timeout=15.0,
    )
    r.raise_for_status()
    return r.json()


def execute_batch_update(
    presentation_id: str, requests: list[dict], access_token: str
) -> dict:
    """Send a batchUpdate to the Slides API."""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    r = httpx.post(
        f"{SLIDES_API}/{presentation_id}:batchUpdate",
        headers=headers,
        json={"requests": requests},
        timeout=30.0,
    )
    r.raise_for_status()
    return r.json()


def gen_id(prefix: str = "dex2") -> str:
    """Generate a short unique objectId valid for Slides API."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def hex_to_rgb(hex_color: str) -> dict:
    """Convert '#RRGGBB' to Slides API rgbColor (0.0-1.0 floats)."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return {"red": 0, "green": 0, "blue": 0}
    return {
        "red": int(h[0:2], 16) / 255.0,
        "green": int(h[2:4], 16) / 255.0,
        "blue": int(h[4:6], 16) / 255.0,
    }


def resolve_insertion_index(
    insert_after, current_slide_index: Optional[int], total_slides: int
) -> int:
    """Compute the 0-based insertion index for a new slide."""
    if insert_after == "end":
        return total_slides
    elif insert_after == "current" and current_slide_index is not None:
        return current_slide_index + 1
    elif isinstance(insert_after, (int, float)):
        return int(insert_after)
    return total_slides
