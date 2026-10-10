"""Recover this deleted workspace from the local Codex patch event history.

This is a one-time recovery aid. It replays only 2026-09 MPC-CNN patches
whose original destination was the deleted rm-project directory.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path


OLD_ROOT = "rm-project"
CUTOFF = "2026-09-29T12:00:00"
HEADER = re.compile(r"\*\*\* (Add|Update|Delete) File: ([^\n]+)")


def source_patches(sessions: Path):
    found = []
    for filename in glob.glob(str(sessions / "2026" / "09" / "**" / "*.jsonl"), recursive=True):
        with open(filename, encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if '"custom_tool_call"' not in line or "tools.apply_patch(" not in line:
                    continue
                try:
                    event = json.loads(line)
                    payload = event["payload"]
                    if payload.get("type") != "custom_tool_call":
                        continue
                    argument = payload.get("input", "")
                except (KeyError, ValueError, TypeError):
                    continue
                timestamp = event.get("timestamp", "")
                if timestamp >= CUTOFF:
                    continue
                # Patches occur both as tools.apply_patch("...") and as
                # const patch = "..."; tools.apply_patch(patch). Scan literals
                # rather than assuming only the first call in a JS cell.
                for match in re.finditer(r'"(?:\\.|[^"\\])*"', argument):
                    if "*** Begin Patch" not in match.group():
                        continue
                    try:
                        patch = json.loads(match.group())
                    except ValueError:
                        continue
                    if "*** Begin Patch" in patch and OLD_ROOT in patch:
                        found.append((timestamp, patch))
    return sorted(set(found))


def destination(root: Path, original: str):
    path = original.replace("\\", "/")
    marker = "/" + OLD_ROOT + "/"
    if marker not in path:
        return None
    relative = path.split(marker, 1)[1]
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"unsafe recovered path {relative!r}")
    return target


def apply_update(path: Path, section: list[str]):
    if not path.exists():
        return "missing update base"
    original = path.read_text(encoding="utf-8")
    content = original.splitlines()
    chunks = []
    current = []
    for line in section:
        if line.startswith("@@"):
            if current:
                chunks.append(current)
            current = []
        else:
            current.append(line)
    if current:
        chunks.append(current)
    cursor = 0
    for chunk in chunks:
        old = [line[1:] for line in chunk if line.startswith((" ", "-"))]
        new = [line[1:] for line in chunk if line.startswith((" ", "+"))]
        if not old:
            content[cursor:cursor] = new
            cursor += len(new)
            continue
        positions = [i for i in range(cursor, len(content) - len(old) + 1)
                     if content[i:i + len(old)] == old]
        if not positions:
            positions = [i for i in range(0, len(content) - len(old) + 1)
                         if content[i:i + len(old)] == old]
        if not positions:
            return f"unmatched hunk: {old[:2]!r}"
        position = positions[0]
        content[position:position + len(old)] = new
        cursor = position + len(new)
    newline = "\n" if original.endswith(("\n", "\r\n")) or content else ""
    path.write_text("\n".join(content) + newline, encoding="utf-8", newline="\n")
    return None


def replay(root: Path, patches):
    applied = []
    warnings = []
    for timestamp, patch in patches:
        lines = patch.splitlines()
        i = 0
        while i < len(lines):
            match = HEADER.fullmatch(lines[i])
            if not match:
                i += 1
                continue
            action, original = match.groups()
            i += 1
            section = []
            while i < len(lines) and not HEADER.fullmatch(lines[i]) and lines[i] != "*** End Patch":
                section.append(lines[i])
                i += 1
            path = destination(root, original)
            if path is None:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            if action == "Add":
                text = "\n".join(line[1:] for line in section if line.startswith("+")) + "\n"
                path.write_text(text, encoding="utf-8", newline="\n")
                issue = None
            elif action == "Delete":
                issue = None if path.exists() else "missing delete target"
                if path.exists():
                    path.unlink()
            else:
                issue = apply_update(path, section)
            entry = f"{timestamp} {action} {path.relative_to(root)}"
            (warnings if issue else applied).append(entry + (f": {issue}" if issue else ""))
    return applied, warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, default=Path.home() / ".codex" / "sessions")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / ".gitignore").write_text(".DS_Store\n__pycache__/\n*.py[cod]\n", encoding="utf-8")
    (root / "README.md").write_text("# rm-project", encoding="utf-8")
    patches = source_patches(args.sessions)
    applied, warnings = replay(root, patches)
    print(json.dumps({"patch_events": len(patches), "applied_sections": len(applied),
                      "warnings": warnings}, indent=2))


if __name__ == "__main__":
    main()
