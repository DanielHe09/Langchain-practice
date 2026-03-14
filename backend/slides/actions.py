"""
Translate LLM instructions into Google Slides API requests and execute them.
"""

from typing import Optional

import httpx

from .api import PT_TO_EMU, execute_batch_update, gen_id, hex_to_rgb


def create_shape(
    inst: dict, presentation_id: str, page_id: str, access_token: str,
) -> str:
    """Create a shape on the specified slide with optional text, fill, border, and styling."""
    obj_id = gen_id("shape")
    shape_type = inst.get("shape_type", "TEXT_BOX")
    x_pt = inst.get("x_pt", 50)
    y_pt = inst.get("y_pt", 50)
    width_pt = inst.get("width_pt", 400)
    height_pt = inst.get("height_pt", 100)
    text = inst.get("text", "")

    requests = [
        {
            "createShape": {
                "objectId": obj_id,
                "shapeType": shape_type,
                "elementProperties": {
                    "pageObjectId": page_id,
                    "size": {
                        "width": {"magnitude": width_pt, "unit": "PT"},
                        "height": {"magnitude": height_pt, "unit": "PT"},
                    },
                    "transform": {
                        "scaleX": 1, "scaleY": 1, "shearX": 0, "shearY": 0,
                        "translateX": x_pt * PT_TO_EMU,
                        "translateY": y_pt * PT_TO_EMU,
                        "unit": "EMU",
                    },
                },
            }
        }
    ]

    shape_props = {}
    shape_fields = []
    if "background_color" in inst:
        shape_props["shapeBackgroundFill"] = {
            "solidFill": {"color": {"rgbColor": hex_to_rgb(inst["background_color"])}}
        }
        shape_fields.append("shapeBackgroundFill.solidFill.color")
    if "border_color" in inst or (inst.get("border_weight_pt", 0) or 0) > 0:
        outline = {}
        outline_fields = []
        if "border_color" in inst:
            outline["outlineFill"] = {
                "solidFill": {"color": {"rgbColor": hex_to_rgb(inst["border_color"])}}
            }
            outline_fields.append("outline.outlineFill.solidFill.color")
        weight_pt = inst.get("border_weight_pt") or 0
        if weight_pt > 0:
            outline["weight"] = {"magnitude": weight_pt, "unit": "PT"}
            outline_fields.append("outline.weight")
        if outline:
            shape_props["outline"] = outline
            shape_fields.extend(outline_fields)
    if shape_props and shape_fields:
        requests.append({
            "updateShapeProperties": {
                "objectId": obj_id,
                "shapeProperties": shape_props,
                "fields": ",".join(shape_fields),
            }
        })

    if text:
        requests.append({
            "insertText": {"objectId": obj_id, "text": text, "insertionIndex": 0}
        })

    text_style = {}
    text_fields = []
    if "font_size_pt" in inst:
        text_style["fontSize"] = {"magnitude": inst["font_size_pt"], "unit": "PT"}
        text_fields.append("fontSize")
    if "bold" in inst:
        text_style["bold"] = inst["bold"]
        text_fields.append("bold")
    if "italic" in inst:
        text_style["italic"] = inst["italic"]
        text_fields.append("italic")
    if "underline" in inst:
        text_style["underline"] = inst["underline"]
        text_fields.append("underline")
    if "font_family" in inst:
        text_style["fontFamily"] = inst["font_family"]
        text_fields.append("fontFamily")
    if "color" in inst:
        text_style["foregroundColor"] = {
            "opaqueColor": {"rgbColor": hex_to_rgb(inst["color"])}
        }
        text_fields.append("foregroundColor")
    if text_style and text_fields and text:
        requests.append({
            "updateTextStyle": {
                "objectId": obj_id,
                "textRange": {"type": "ALL"},
                "style": text_style,
                "fields": ",".join(text_fields),
            }
        })

    execute_batch_update(presentation_id, requests, access_token)
    label = shape_type.lower().replace("_", " ")
    snippet = f"'{text[:40]}...'" if len(text) > 40 else (f"'{text}'" if text else "(empty)")
    return f"Created {label} {snippet} at ({x_pt}, {y_pt})"


def create_line(
    inst: dict, presentation_id: str, page_id: str, access_token: str,
) -> str:
    """Create a line on the specified slide."""
    obj_id = gen_id("line")
    line_type = inst.get("line_type", "STRAIGHT")
    category_map = {"STRAIGHT": "STRAIGHT", "BENT": "BENT", "CURVED": "CURVED"}
    category = category_map.get(line_type, "STRAIGHT")

    start_x = inst.get("start_x_pt", 0)
    start_y = inst.get("start_y_pt", 0)
    end_x = inst.get("end_x_pt", 100)
    end_y = inst.get("end_y_pt", 100)

    width_pt = abs(end_x - start_x) or 1
    height_pt = abs(end_y - start_y) or 1
    x_pt = min(start_x, end_x)
    y_pt = min(start_y, end_y)

    requests = [
        {
            "createLine": {
                "objectId": obj_id,
                "lineCategory": category,
                "elementProperties": {
                    "pageObjectId": page_id,
                    "size": {
                        "width": {"magnitude": width_pt, "unit": "PT"},
                        "height": {"magnitude": height_pt, "unit": "PT"},
                    },
                    "transform": {
                        "scaleX": 1, "scaleY": 1, "shearX": 0, "shearY": 0,
                        "translateX": x_pt * PT_TO_EMU,
                        "translateY": y_pt * PT_TO_EMU,
                        "unit": "EMU",
                    },
                },
            }
        }
    ]

    line_props = {}
    line_fields = []
    if "color" in inst:
        line_props["lineFill"] = {
            "solidFill": {"color": {"rgbColor": hex_to_rgb(inst["color"])}}
        }
        line_fields.append("lineFill.solidFill.color")
    if "weight_pt" in inst:
        line_props["weight"] = {"magnitude": inst["weight_pt"], "unit": "PT"}
        line_fields.append("weight")
    if line_props and line_fields:
        requests.append({
            "updateLineProperties": {
                "objectId": obj_id,
                "lineProperties": line_props,
                "fields": ",".join(line_fields),
            }
        })

    execute_batch_update(presentation_id, requests, access_token)
    return f"Created {line_type.lower()} line from ({start_x}, {start_y}) to ({end_x}, {end_y})"


def edit_instructions_to_batch_requests(
    instructions: list[dict], page_json: dict,
) -> list[dict]:
    """
    Convert LLM instructions (move/resize/replace_text/update_text_style)
    to Slides API batchUpdate requests.
    Skips creation instructions (handled separately).
    """
    el_map = {}
    for el in page_json.get("pageElements", []):
        el_map[el["objectId"]] = el

    requests = []
    for inst in instructions:
        action = inst.get("action")

        if action in ("create_slide", "create_shape", "create_line"):
            continue

        obj_id = inst.get("objectId")
        if not obj_id or obj_id not in el_map:
            continue

        el = el_map[obj_id]
        has_text = bool(el.get("shape", {}).get("text", {}).get("textElements"))

        if action == "replace_text":
            new_text = inst.get("new_text", "")
            if has_text:
                requests.append({
                    "deleteText": {"objectId": obj_id, "textRange": {"type": "ALL"}}
                })
            requests.append({
                "insertText": {"objectId": obj_id, "text": new_text, "insertionIndex": 0}
            })
            continue

        if action == "update_text_style":
            if not has_text:
                print(f"   SLIDES: skipping update_text_style for {obj_id} (no text)")
                continue
            style = {}
            fields = []
            if "font_size_pt" in inst:
                style["fontSize"] = {"magnitude": inst["font_size_pt"], "unit": "PT"}
                fields.append("fontSize")
            if "bold" in inst:
                style["bold"] = inst["bold"]
                fields.append("bold")
            if "italic" in inst:
                style["italic"] = inst["italic"]
                fields.append("italic")
            if "underline" in inst:
                style["underline"] = inst["underline"]
                fields.append("underline")
            if "font_family" in inst:
                style["fontFamily"] = inst["font_family"]
                fields.append("fontFamily")
            if "color" in inst:
                style["foregroundColor"] = {
                    "opaqueColor": {"rgbColor": hex_to_rgb(inst["color"])}
                }
                fields.append("foregroundColor")
            if style and fields:
                requests.append({
                    "updateTextStyle": {
                        "objectId": obj_id,
                        "textRange": {"type": "ALL"},
                        "style": style,
                        "fields": ",".join(fields),
                    }
                })
            continue

        if action == "update_shape_fill":
            if "shape" not in el:
                print(f"   SLIDES: skipping update_shape_fill for {obj_id} (not a shape)")
                continue
            shape_props = {}
            shape_fields = []
            if "background_color" in inst:
                shape_props["shapeBackgroundFill"] = {
                    "solidFill": {"color": {"rgbColor": hex_to_rgb(inst["background_color"])}}
                }
                shape_fields.append("shapeBackgroundFill.solidFill.color")
            weight_pt = inst.get("border_weight_pt") or 0
            if "border_color" in inst or weight_pt > 0:
                outline = {}
                outline_fields = []
                if "border_color" in inst:
                    outline["outlineFill"] = {
                        "solidFill": {"color": {"rgbColor": hex_to_rgb(inst["border_color"])}}
                    }
                    outline_fields.append("outline.outlineFill.solidFill.color")
                if weight_pt > 0:
                    outline["weight"] = {"magnitude": weight_pt, "unit": "PT"}
                    outline_fields.append("outline.weight")
                shape_props["outline"] = outline
                shape_fields.extend(outline_fields)
            if shape_props and shape_fields:
                requests.append({
                    "updateShapeProperties": {
                        "objectId": obj_id,
                        "shapeProperties": shape_props,
                        "fields": ",".join(shape_fields),
                    }
                })
            continue

        # Layout instructions (move / resize / move_and_resize)
        transform = el.get("transform", {})
        size = el.get("size", {})

        existing_sx = transform.get("scaleX", 1)
        existing_sy = transform.get("scaleY", 1)
        existing_shx = transform.get("shearX", 0)
        existing_shy = transform.get("shearY", 0)
        existing_tx = transform.get("translateX", 0)
        existing_ty = transform.get("translateY", 0)

        new_tx = existing_tx
        new_ty = existing_ty
        new_sx = existing_sx
        new_sy = existing_sy

        if action in ("move", "move_and_resize"):
            if "x_pt" in inst:
                new_tx = inst["x_pt"] * PT_TO_EMU
            if "y_pt" in inst:
                new_ty = inst["y_pt"] * PT_TO_EMU

        if action in ("resize", "move_and_resize"):
            w_dim = size.get("width", {})
            h_dim = size.get("height", {})
            raw_w = w_dim.get("magnitude", 1)
            raw_h = h_dim.get("magnitude", 1)
            raw_w_pt = raw_w if w_dim.get("unit") == "PT" else raw_w / PT_TO_EMU
            raw_h_pt = raw_h if h_dim.get("unit") == "PT" else raw_h / PT_TO_EMU

            if "width_pt" in inst and raw_w_pt > 0:
                new_sx = inst["width_pt"] / raw_w_pt
            if "height_pt" in inst and raw_h_pt > 0:
                new_sy = inst["height_pt"] / raw_h_pt

        requests.append({
            "updatePageElementTransform": {
                "objectId": obj_id,
                "applyMode": "ABSOLUTE",
                "transform": {
                    "scaleX": new_sx, "scaleY": new_sy,
                    "shearX": existing_shx, "shearY": existing_shy,
                    "translateX": new_tx, "translateY": new_ty,
                    "unit": "EMU",
                },
            }
        })

    return requests


def apply_instructions(
    instructions: list[dict],
    presentation_id: str,
    page_id: str,
    page_json: dict,
    access_token: str,
) -> tuple[int, int, Optional[str]]:
    """
    Apply a list of instructions. Returns (shapes_created, elements_updated, error_msg).
    error_msg is None on success.
    """
    shapes_created = 0
    elements_updated = 0

    for si in [i for i in instructions if i.get("action") == "create_shape"]:
        try:
            result = create_shape(si, presentation_id, page_id, access_token)
            print(f"   SLIDES: {result}")
            shapes_created += 1
        except httpx.HTTPStatusError as e:
            body = e.response.text[:300] if e.response else ""
            print(f"   SLIDES: create_shape failed (HTTP {e.response.status_code}): {body}")
            return shapes_created, elements_updated, f"Failed to create shape (HTTP {e.response.status_code}). Error: {body}"
        except Exception as e:
            return shapes_created, elements_updated, f"Error creating shape: {e}"

    for li in [i for i in instructions if i.get("action") == "create_line"]:
        try:
            result = create_line(li, presentation_id, page_id, access_token)
            print(f"   SLIDES: {result}")
            shapes_created += 1
        except httpx.HTTPStatusError as e:
            body = e.response.text[:300] if e.response else ""
            return shapes_created, elements_updated, f"Failed to create line (HTTP {e.response.status_code}). Error: {body}"
        except Exception as e:
            return shapes_created, elements_updated, f"Error creating line: {e}"

    layout_requests = edit_instructions_to_batch_requests(instructions, page_json)
    if layout_requests:
        try:
            execute_batch_update(presentation_id, layout_requests, access_token)
            elements_updated = len(layout_requests)
        except httpx.HTTPStatusError as e:
            body = e.response.text[:300] if e.response else ""
            return shapes_created, elements_updated, f"Failed to update elements (HTTP {e.response.status_code})."
        except Exception as e:
            return shapes_created, elements_updated, f"Error updating elements: {e}"

    return shapes_created, elements_updated, None
