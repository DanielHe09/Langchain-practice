"""
Build human-readable slide/presentation descriptions for LLM context.
"""

from typing import Optional

from .api import PT_TO_EMU


def _summarize_element(el: dict) -> Optional[dict]:
    """
    Summarize a page element into a readable dict for the LLM.
    Returns None if the element has no transform (not movable).
    """
    transform = el.get("transform")
    size_obj = el.get("size", {})
    if not transform:
        return None

    w_dim = size_obj.get("width", {})
    h_dim = size_obj.get("height", {})
    w_mag = w_dim.get("magnitude", 0)
    h_mag = h_dim.get("magnitude", 0)
    w_unit = w_dim.get("unit", "EMU")
    h_unit = h_dim.get("unit", "EMU")

    width_pt = (w_mag / PT_TO_EMU) if w_unit == "EMU" else w_mag
    height_pt = (h_mag / PT_TO_EMU) if h_unit == "EMU" else h_mag

    scale_x = transform.get("scaleX", 1)
    scale_y = transform.get("scaleY", 1)
    tx = transform.get("translateX", 0)
    ty = transform.get("translateY", 0)

    rendered_w = width_pt * abs(scale_x)
    rendered_h = height_pt * abs(scale_y)
    x_pt = tx / PT_TO_EMU
    y_pt = ty / PT_TO_EMU

    el_type = "shape"
    text_content = ""
    text_style_desc = ""
    if "shape" in el:
        shape_type = el["shape"].get("shapeType", "RECTANGLE")
        if shape_type == "TEXT_BOX":
            el_type = "text_box"
        else:
            el_type = f"shape ({shape_type})"
        text_runs = []
        styles_seen = []
        for te in el["shape"].get("text", {}).get("textElements", []):
            run = te.get("textRun", {})
            content = run.get("content", "").strip()
            if content:
                text_runs.append(content)
            style = run.get("style", {})
            if style and content:
                s_parts = []
                font = style.get("fontFamily")
                if font:
                    s_parts.append(font)
                fs = style.get("fontSize", {})
                if fs.get("magnitude"):
                    s_parts.append(f"{fs['magnitude']}{fs.get('unit', 'PT').lower()}")
                if style.get("bold"):
                    s_parts.append("bold")
                if style.get("italic"):
                    s_parts.append("italic")
                fg = style.get("foregroundColor", {}).get("opaqueColor", {}).get("rgbColor", {})
                if fg:
                    r_val = int(fg.get("red", 0) * 255)
                    g_val = int(fg.get("green", 0) * 255)
                    b_val = int(fg.get("blue", 0) * 255)
                    hex_color = f"#{r_val:02x}{g_val:02x}{b_val:02x}"
                    if hex_color != "#000000":
                        s_parts.append(hex_color)
                if s_parts:
                    styles_seen.append(" ".join(s_parts))
        text_content = " ".join(text_runs)
        if styles_seen:
            unique_styles = list(dict.fromkeys(styles_seen))
            text_style_desc = " | ".join(unique_styles[:3])
        bg_fill = el["shape"].get("shapeProperties", {}).get("shapeBackgroundFill", {})
        solid = bg_fill.get("solidFill", {})
        bg_rgb = solid.get("color", {}).get("rgbColor", {})
        if bg_rgb:
            br = int(bg_rgb.get("red", 1) * 255)
            bg = int(bg_rgb.get("green", 1) * 255)
            bb = int(bg_rgb.get("blue", 1) * 255)
            bg_hex = f"#{br:02x}{bg:02x}{bb:02x}"
            if bg_hex != "#ffffff":
                text_style_desc += f" | bg:{bg_hex}"
    elif "image" in el:
        el_type = "image"
    elif "table" in el:
        el_type = "table"
    elif "elementGroup" in el:
        el_type = "group"

    summary = {
        "objectId": el["objectId"],
        "type": el_type,
        "x_pt": round(x_pt, 1),
        "y_pt": round(y_pt, 1),
        "width_pt": round(rendered_w, 1),
        "height_pt": round(rendered_h, 1),
    }
    if text_content:
        summary["text"] = text_content[:100]
    if text_style_desc:
        summary["style"] = text_style_desc
    return summary


def _compute_free_gaps(
    summaries: list[dict], page_h_pt: float
) -> list[tuple[float, float, float]]:
    """Compute vertical free-space gaps from element summaries."""
    occupied = []
    for s in summaries:
        if s["width_pt"] > 0 and s["height_pt"] > 0:
            occupied.append((s["y_pt"], s["y_pt"] + s["height_pt"]))
    occupied.sort()
    merged = []
    for start, end in occupied:
        if merged and start <= merged[-1][1] + 2:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    gaps = []
    prev_end = 0.0
    for start, end in merged:
        if start - prev_end >= 15:
            gaps.append((round(prev_end, 1), round(start, 1), round(start - prev_end, 1)))
        prev_end = end
    if page_h_pt - prev_end >= 15:
        gaps.append((round(prev_end, 1), round(page_h_pt, 1), round(page_h_pt - prev_end, 1)))
    return gaps


def build_slide_description(
    page_json: dict, page_width_emu: int, page_height_emu: int
) -> tuple[str, list[tuple[float, float, float]], int]:
    """Build a human-readable description of a slide. Returns (description, gaps, num_elements)."""
    page_w_pt = round(page_width_emu / PT_TO_EMU, 1)
    page_h_pt = round(page_height_emu / PT_TO_EMU, 1)

    elements = page_json.get("pageElements", [])
    summaries = []
    for el in elements:
        s = _summarize_element(el)
        if s:
            summaries.append(s)

    gaps = _compute_free_gaps(summaries, page_h_pt)

    lines = [
        f"Slide dimensions: {page_w_pt} x {page_h_pt} PT (points). Origin (0,0) is top-left.",
        f"Number of elements: {len(summaries)}",
    ]

    if gaps:
        lines.append("")
        lines.append("*** FREE SPACE — place new elements ONLY in these vertical gaps: ***")
        for g_start, g_end, g_height in gaps:
            lines.append(f"  GAP: y={g_start} to y={g_end} (max height: {g_height} PT)")
        lines.append("  A new element at y=Y with height=H MUST satisfy: Y >= gap_start AND Y + H <= gap_end.")
        lines.append("  If your content doesn't fit, SHRINK it (reduce height or font size) to fit within the gap.")

    lines.append("")
    lines.append("Elements:")
    for s in summaries:
        desc = f'  - objectId: "{s["objectId"]}", type: {s["type"]}, position: ({s["x_pt"]}, {s["y_pt"]}) PT, size: {s["width_pt"]} x {s["height_pt"]} PT'
        if s.get("text"):
            desc += f', text: "{s["text"]}"'
        if s.get("style"):
            desc += f', style: [{s["style"]}]'
        lines.append(desc)

    return "\n".join(lines), gaps, len(summaries)


def _summarize_slide_brief(slide: dict, index: int) -> str:
    """Short summary of a slide (for non-current slides), including text style."""
    elements = slide.get("pageElements", [])
    texts = []
    styles = []
    for el in elements:
        for te in el.get("shape", {}).get("text", {}).get("textElements", []):
            run = te.get("textRun", {})
            t = run.get("content", "").strip()
            if t:
                texts.append(t)
            style = run.get("style", {})
            if style and t:
                s_parts = []
                font = style.get("fontFamily")
                if font:
                    s_parts.append(font)
                fs = style.get("fontSize", {})
                if fs.get("magnitude"):
                    s_parts.append(f"{fs['magnitude']}{fs.get('unit', 'PT').lower()}")
                if style.get("bold"):
                    s_parts.append("bold")
                if s_parts:
                    styles.append(" ".join(s_parts))
    combined = " | ".join(texts)
    if len(combined) > 150:
        combined = combined[:150] + "..."
    el_count = len(elements)
    unique_styles = list(dict.fromkeys(styles))
    style_str = f" (style: {', '.join(unique_styles[:2])})" if unique_styles else ""
    if combined:
        return f"  Slide {index} ({el_count} elements): \"{combined}\"{style_str}"
    return f"  Slide {index} ({el_count} elements): [no text]{style_str}"


def build_full_presentation_context(
    presentation: dict, current_page_id: Optional[str],
    page_width_emu: int, page_height_emu: int,
) -> tuple[str, Optional[dict], int, Optional[int], list, int]:
    """
    Build context string with ALL slides summarized.
    Returns (context_string, current_page_json, total_slides, current_slide_index, gaps, num_elements).
    """
    slides = presentation.get("slides", [])
    total = len(slides)
    current_index = None
    current_page_json = None

    for i, s in enumerate(slides):
        if s.get("objectId") == current_page_id:
            current_index = i
            current_page_json = s
            break

    other_lines = []
    for i, s in enumerate(slides):
        if i == current_index:
            continue
        other_lines.append(_summarize_slide_brief(s, i))

    gaps = []
    num_elements = 0
    if current_page_json:
        current_desc, gaps, num_elements = build_slide_description(
            current_page_json, page_width_emu, page_height_emu
        )
    else:
        current_desc = "(Could not find the current slide)"

    lines = [
        f"=== PRESENTATION ({total} slides) ===",
        f"Title: {presentation.get('title', 'Untitled')}",
        "",
        f"=== CURRENT SLIDE (index {current_index}, the slide the user is viewing) ===",
        current_desc,
        "",
        f"=== OTHER SLIDES (summary) ===",
    ]
    lines.extend(other_lines if other_lines else ["  (no other slides)"])

    return "\n".join(lines), current_page_json, total, current_index, gaps, num_elements


def extract_presentation_style(presentation: dict) -> str:
    """Extract dominant visual style patterns from the presentation."""
    fonts: dict[str, int] = {}
    bg_colors: dict[str, int] = {}
    text_colors: dict[str, int] = {}
    for slide in presentation.get("slides", []):
        for el in slide.get("pageElements", []):
            shape = el.get("shape", {})
            bg_fill = shape.get("shapeProperties", {}).get("shapeBackgroundFill", {})
            solid = bg_fill.get("solidFill", {})
            bg_rgb = solid.get("color", {}).get("rgbColor", {})
            if bg_rgb:
                r = int(bg_rgb.get("red", 1) * 255)
                g = int(bg_rgb.get("green", 1) * 255)
                b = int(bg_rgb.get("blue", 1) * 255)
                h = f"#{r:02x}{g:02x}{b:02x}"
                if h != "#ffffff":
                    bg_colors[h] = bg_colors.get(h, 0) + 1
            for te in shape.get("text", {}).get("textElements", []):
                run = te.get("textRun", {})
                style = run.get("style", {})
                if not run.get("content", "").strip():
                    continue
                font = style.get("fontFamily")
                if font:
                    fonts[font] = fonts.get(font, 0) + 1
                fg = style.get("foregroundColor", {}).get("opaqueColor", {}).get("rgbColor", {})
                if fg:
                    r = int(fg.get("red", 0) * 255)
                    g = int(fg.get("green", 0) * 255)
                    b = int(fg.get("blue", 0) * 255)
                    h = f"#{r:02x}{g:02x}{b:02x}"
                    if h != "#000000":
                        text_colors[h] = text_colors.get(h, 0) + 1

    lines = ["Presentation visual style:"]
    top_fonts = sorted(fonts, key=fonts.get, reverse=True)[:3]
    if top_fonts:
        lines.append(f"  Common fonts: {', '.join(top_fonts)}")
    top_bg = sorted(bg_colors, key=bg_colors.get, reverse=True)[:4]
    if top_bg:
        lines.append(f"  Common background fills: {', '.join(top_bg)}")
    top_tc = sorted(text_colors, key=text_colors.get, reverse=True)[:3]
    if top_tc:
        lines.append(f"  Common text accent colors: {', '.join(top_tc)}")
    return "\n".join(lines)
