#!/usr/bin/env python3
import json
import sys
from pathlib import Path

ROLE_TITLE = {
    "system": "System",
    "user": "User",
    "assistant": "Assistant",
    "tool": "Tool",
}

def flatten_content(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        parts = [flatten_content(i) for i in x]
        return "\n".join(p for p in parts if p.strip())
    if isinstance(x, dict):
        for key in ("text", "output_text", "input_text", "value"):
            val = x.get(key)
            if isinstance(val, str):
                return val
        if "content" in x:
            return flatten_content(x["content"])
        if "message" in x:
            return flatten_content(x["message"])
        if "arguments" in x:
            return "```json\n" + str(x["arguments"]) + "\n```"
    return ""

def extract_message(obj):
    candidates = [
        obj,
        obj.get("message"),
        obj.get("payload"),
        obj.get("data"),
        obj.get("item"),
    ]
    for c in candidates:
        if not isinstance(c, dict):
            continue
        if c.get("role") and c.get("content") is not None:
            return c["role"], flatten_content(c["content"])
        if c.get("type") == "message" and c.get("role"):
            return c["role"], flatten_content(c.get("content"))
    return None

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} session.jsonl", file=sys.stderr)
        sys.exit(1)

    path = Path(sys.argv[1]).expanduser()
    out = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msg = extract_message(obj)
            if not msg:
                continue
            role, text = msg
            text = text.strip()
            if not text:
                continue
            title = ROLE_TITLE.get(role, role.title())
            out.append(f"## {title}\n\n{text}\n")

    print("\n".join(out).strip())

if __name__ == "__main__":
    main()