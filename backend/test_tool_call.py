"""Minimal test: LLM + bind_tools with 'open youtube' to verify tool_calls."""
import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

MAIN_AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "open_tab",
            "description": "Open a new browser tab with the given URL. Use when the user wants to open a website, search for something, or go to a URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL to open (must start with https:// or http://)"},
                    "message": {"type": "string", "description": "Short user-facing message."},
                },
                "required": ["url"],
            },
        },
    },
]

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set, skipping live call")
        return
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    bound = llm.bind_tools(MAIN_AGENT_TOOLS)
    messages = [
        SystemMessage(content="When the user wants to open a site, use open_tab with the full URL. Otherwise answer in text."),
        HumanMessage(content="Open youtube.com for me"),
    ]
    response = bound.invoke(messages)
    tool_calls = getattr(response, "tool_calls", None) or []
    print("tool_calls count:", len(tool_calls))
    for tc in tool_calls:
        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
        args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
        print("  name:", name, "args:", json.dumps(args))
    print("content:", (response.content or "")[:200])
    if tool_calls and (tool_calls[0].get("name") if isinstance(tool_calls[0], dict) else getattr(tool_calls[0], "name", "")) == "open_tab":
        print("OK: open_tab tool was requested")
    else:
        print("Note: model did not call open_tab (may still be OK if content is set)")

if __name__ == "__main__":
    main()
