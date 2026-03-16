"""
Extract slide style (colors, font) from a slide screenshot using Gemini vision.
Used when API-based style (get_presentation_style_values) is missing or unreliable.
"""

import base64
import json
import os
import re
from typing import Any, Optional

import google.generativeai as genai


STYLE_PROMPT = """Look at this screenshot of a single slide from a presentation.

Extract the dominant visual style used for text boxes and shapes on the slide. Reply with a single JSON object (no markdown, no code fence) with exactly these keys:
- "primary_text_color": color of the text inside text boxes, as hex (e.g. "#ffffff" or "#333333")
- "primary_font": main font family name used for that text (e.g. "Roboto", "Playfair Display", "Arial")
- "primary_background_fills": list of 1–3 hex colors used as the background fill inside text boxes and shapes (e.g. the fill color you see behind the text). Use the slide background color if text boxes match it (e.g. ["#21b2dd"]).
- "primary_border_colors": list of 1–2 hex colors used for the outline/border of text boxes and shapes (e.g. ["#000000"] for black borders)

Identify the actual fill and outline colors you see on the text boxes in the image. If you cannot determine a value, use: text "#333333", font "Arial", background ["#ffffff"], border ["#000000"].
Output only the JSON object, nothing else."""


def _decode_screenshot_data(screenshot_base64: str) -> tuple[bytes, str]:
    """Return (image_bytes, mime_type). Handles raw base64 or data URL."""
    data = (screenshot_base64 or "").strip()
    if data.startswith("data:"):
        match = re.match(r"data:([^;]+);base64,(.+)", data, re.DOTALL)
        if match:
            mime = match.group(1).strip().lower()
            if "png" in mime:
                mime = "image/png"
            elif "jpeg" in mime or "jpg" in mime:
                mime = "image/jpeg"
            else:
                mime = "image/png"
            return base64.b64decode(match.group(2)), mime
        data = data.split(",", 1)[-1]
    return base64.b64decode(data), "image/png"


def extract_style_from_slide_image(screenshot_base64: Optional[str]) -> Optional[dict[str, Any]]:
    """
    Use Gemini vision to infer primary text color, font, background fills, and border colors
    from a slide screenshot. Returns a dict compatible with normalize_instructions_style:
    primary_font, primary_text_color, primary_background_fills, primary_border_colors.
    Returns None on parse/API error.
    """
    if not screenshot_base64 or not screenshot_base64.strip():
        return None
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("   VISION_STYLE: GOOGLE_API_KEY not set, skipping vision style")
        return None
    try:
        image_bytes, mime_type = _decode_screenshot_data(screenshot_base64)
    except Exception as e:
        print(f"   VISION_STYLE: failed to decode screenshot: {e}")
        return None
    genai.configure(api_key=api_key)
    # Use stable vision-capable model (gemini-1.5-flash is deprecated/removed)
    model = genai.GenerativeModel("gemini-2.5-flash")
    parts = [
        {"inline_data": {"mime_type": mime_type, "data": image_bytes}},
        STYLE_PROMPT,
    ]
    try:
        response = model.generate_content(parts)
        text = (response.text or "").strip()
    except Exception as e:
        print(f"   VISION_STYLE: Gemini API error: {e}")
        return None
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    text = text.strip()
    try:
        out = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"   VISION_STYLE: JSON parse error: {e}, raw={text[:200]!r}")
        return None
    primary_font = out.get("primary_font") or "Arial"
    primary_text_color = out.get("primary_text_color") or "#333333"
    primary_background_fills = out.get("primary_background_fills")
    if not isinstance(primary_background_fills, list):
        primary_background_fills = [primary_text_color] if primary_text_color else ["#ffffff"]
    primary_border_colors = out.get("primary_border_colors")
    if not isinstance(primary_border_colors, list):
        primary_border_colors = ["#000000"]
    result = {
        "primary_font": primary_font,
        "primary_text_color": primary_text_color,
        "primary_background_fills": primary_background_fills[:6],
        "primary_border_colors": primary_border_colors[:4],
    }
    print(
        f"   VISION_STYLE: primary_font={primary_font!r} primary_text_color={primary_text_color!r} "
        f"primary_background_fills={primary_background_fills!r} primary_border_colors={primary_border_colors!r}"
    )
    return result
