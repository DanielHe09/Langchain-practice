"""
Executor prompts and LLM call helper — one focused prompt per operation type.
"""

import json
import re

from langchain_agent import llm
from langchain_core.messages import HumanMessage, SystemMessage


# ---------------------------------------------------------------------------
EDIT_LAYOUT_PROMPT = """You reposition and resize elements on a Google Slides slide. Output a JSON object:
{"instructions": [...], "message": "brief summary"}

Each instruction: {"action": "move"|"resize"|"move_and_resize", "objectId": "...", "x_pt": ..., "y_pt": ..., "width_pt": ..., "height_pt": ...}
- "move": set x_pt and/or y_pt
- "resize": set width_pt and/or height_pt
- "move_and_resize": set any combination

Rules:
- Positions from top-left (0,0). Units: PT (points).
- Only include elements you want to change.
- Slide center X = half slide width, center Y = half slide height.
- For symmetry: mirror positions relative to slide center, account for element sizes.
- No markdown fences. Output ONLY the JSON object."""


# ---------------------------------------------------------------------------
CREATE_CONTENT_PROMPT = """You add new shapes, text boxes, and lines to a Google Slides slide. Output a JSON object:
{"instructions": [...], "message": "brief summary"}

Shape instruction:
{"action": "create_shape", "shape_type": "TEXT_BOX"|"RECTANGLE"|"ROUND_RECTANGLE"|"ELLIPSE"|"TRIANGLE"|etc., "x_pt": ..., "y_pt": ..., "width_pt": ..., "height_pt": ..., "text": "...", "background_color": "#hex", "border_color": "#hex", "border_weight_pt": ..., "font_size_pt": ..., "bold": true/false, "italic": true/false, "font_family": "...", "color": "#hex"}

Line instruction:
{"action": "create_line", "line_type": "STRAIGHT"|"BENT"|"CURVED", "start_x_pt": ..., "start_y_pt": ..., "end_x_pt": ..., "end_y_pt": ..., "color": "#hex", "weight_pt": ...}

Rules:
- ALWAYS include "text" with actual content when user asks to add a section/content about something.
- For background_color and color (text): use ONLY hex values from existing elements (style and bg: in the element list) or from "Common background fills" / "Common text accent colors" in the presentation. Do not use #ffffff or generic gray unless the deck already uses them.
- Use RECTANGLE or ROUND_RECTANGLE with background_color for card-style sections.
- A new element at y=Y with height=H MUST fit within a free-space gap: Y >= gap_start AND Y + H <= gap_end.
- If content doesn't fit, SHRINK height or font size to fit within the gap.
- You can also move/resize existing elements to make room: {"action": "move"|"resize"|"move_and_resize", "objectId": "...", ...}
- No markdown fences. Output ONLY the JSON object."""


# ---------------------------------------------------------------------------
CREATE_SLIDE_PROMPT = """You design a new Google Slides slide from scratch using shapes. The slide will be created as BLANK, then you populate it with shapes that match the presentation's visual style.

Output a JSON object:
{"instructions": [...], "message": "brief summary", "insert_after": "current"|"end"|<index>}

Each instruction creates a shape on the NEW slide:
{"action": "create_shape", "shape_type": "TEXT_BOX"|"RECTANGLE"|"ROUND_RECTANGLE"|etc., "x_pt": ..., "y_pt": ..., "width_pt": ..., "height_pt": ..., "text": "...", "background_color": "#hex", "border_color": "#hex", "border_weight_pt": ..., "font_size_pt": ..., "bold": true/false, "italic": true/false, "font_family": "...", "color": "#hex"}

Rules:
- Design a COMPLETE slide layout — add title, body content, decorative elements as needed.
- For every shape, set background_color and color (text) using ONLY the hex values from "Common background fills" and "Common text accent colors" / "Common fonts" in the presentation visual style. Do not invent colors; do not use #ffffff or generic gray unless the deck uses them.
- Cover the full slide area (dimensions provided). Don't leave the slide empty.
- Generate real, substantive content for the topic the user requested.
- Use shapes with background_color for card-style sections, matching existing bg: colors.

COPY FROM CURRENT SLIDE when the user asks to copy the heading, footer, or table of contents:
- HEADING: Look at the CURRENT SLIDE elements at the top. If there is a vertical line + title + subtitle layout, REPLICATE it: create a thin vertical RECTANGLE (same color as original), then a title shape, then a subtitle shape — same relative positions (x_pt, y_pt) and sizes (width_pt, height_pt), same fonts and colors; only change the title/subtitle text to fit the new slide topic.
- TABLE OF CONTENTS / FOOTER: Look at the CURRENT SLIDE elements at the bottom. If there are multiple separate shapes (tabs like "Business Overview", "Market Feasibility", etc.), REPLICATE that structure: create one create_shape per tab, same count, same relative positions and sizes, same bg: and style; you may update the tab labels to match the new slide (e.g. keep "Conclusion" or add "Long Term Trends") but keep the same visual layout.
- Do not replace the heading or ToC with a single simplified shape — copy the actual structure (multiple elements) so it looks like the original.

- No markdown fences. Output ONLY the JSON object."""


# ---------------------------------------------------------------------------
EDIT_TEXT_PROMPT = """You edit text content, formatting, and shape fill/outline on existing Google Slides elements. Output a JSON object:
{"instructions": [...], "message": "brief summary"}

Replace text: {"action": "replace_text", "objectId": "...", "new_text": "..."}
Update text style: {"action": "update_text_style", "objectId": "...", "font_size_pt": ..., "bold": true/false, "italic": true/false, "underline": true/false, "font_family": "...", "color": "#hex"}
Update shape fill/background (for text boxes and shapes): {"action": "update_shape_fill", "objectId": "...", "background_color": "#hex", "border_color": "#hex", "border_weight_pt": number (optional, omit or use 0 for no border)}

Rules:
- replace_text replaces ALL text in the element.
- update_text_style applies to ALL text in the element (color = text color).
- update_shape_fill changes the shape's background fill and/or border. Use when the user says "change the background/fill color", "match the colors", "make these boxes the same color as the rest of the slideshow". Use hex values from the slide's existing elements (style and bg: in the element list) or from "Common background fills" / "Common text accent colors" in the presentation style.
- You can combine replace_text, update_text_style, and update_shape_fill on the same or different elements.
- When the user wants colors to match the slideshow, apply update_shape_fill (and update_text_style for text color) to the relevant elements using the deck's actual colors from the context.
- No markdown fences. Output ONLY the JSON object."""


# ---------------------------------------------------------------------------
ANSWER_QUESTION_PROMPT = """You answer questions about a Google Slides presentation. Output a JSON object:
{"instructions": [], "message": "your detailed answer"}

Rules:
- Always return an empty instructions array.
- Put your answer in "message".
- Use the slide data provided to give accurate, specific answers about fonts, colors, content, structure, etc.
- No markdown fences. Output ONLY the JSON object."""


# ---------------------------------------------------------------------------
EXECUTOR_PROMPTS = {
    "edit_layout": EDIT_LAYOUT_PROMPT,
    "create_content": CREATE_CONTENT_PROMPT,
    "create_slide": CREATE_SLIDE_PROMPT,
    "edit_text": EDIT_TEXT_PROMPT,
    "answer_question": ANSWER_QUESTION_PROMPT,
}


def call_executor(
    system_prompt: str, context: str, user_message: str
) -> tuple[list[dict], str, dict]:
    """Call an executor LLM with a focused prompt. Returns (instructions, message, full_parsed_dict)."""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"{context}\n\nUser request: {user_message}"),
    ]
    response = llm.invoke(messages)
    text = response.content if hasattr(response, "content") else str(response)
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()
    print(f"   EXECUTOR raw: {text[:500]}")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            instructions = parsed.get("instructions", [])
            if not isinstance(instructions, list):
                instructions = []
            return instructions, parsed.get("message", ""), parsed
        if isinstance(parsed, list):
            return parsed, "", {}
        return [], "", {}
    except json.JSONDecodeError as e:
        print(f"   EXECUTOR JSON parse error: {e}")
        return [], "", {}
