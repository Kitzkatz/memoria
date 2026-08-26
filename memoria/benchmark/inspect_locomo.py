#!/usr/bin/env python3
"""
Inspect LoCoMo dataset structure.
"""

import json
import sys
from pathlib import Path

def inspect_value(value, depth=0, max_depth=3):
    """Pretty‑print a value with type and snippet."""
    indent = "  " * depth
    if isinstance(value, dict):
        print(f"{indent}{{dict with {len(value)} keys}}")
        if depth < max_depth:
            for k, v in list(value.items())[:5]:
                print(f"{indent}  {k!r}: ", end="")
                inspect_value(v, depth+1, max_depth)
            if len(value) > 5:
                print(f"{indent}  ... and {len(value)-5} more keys")
    elif isinstance(value, list):
        print(f"{indent}[list of {len(value)} items]")
        if depth < max_depth and value:
            print(f"{indent}  first item:")
            inspect_value(value[0], depth+1, max_depth)
    elif isinstance(value, str):
        snippet = value[:60] + ("..." if len(value) > 60 else "")
        print(f"{indent}{snippet!r}")
    else:
        print(f"{indent}{value!r}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="locomo10.json", help="Path to LoCoMo JSON")
    parser.add_argument("--index", type=int, default=0, help="Entry index to inspect")
    args = parser.parse_args()

    path = Path(args.dataset)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not data:
        print("Dataset is empty.")
        sys.exit(0)

    entry = data[args.index] if args.index < len(data) else data[0]
    print(f"Inspecting entry {args.index} (total {len(data)} entries)")
    print("=" * 60)

    # Main structure
    inspect_value(entry, max_depth=4)

    # Also show a few QA items if present
    qa = entry.get("qa")
    if qa and isinstance(qa, list):
        print("\n--- Sample QA item ---")
        if qa:
            inspect_value(qa[0], max_depth=3)

    # Show conversation sessions if present
    conv = entry.get("conversation")
    if conv and isinstance(conv, dict):
        print("\n--- Conversation sessions ---")
        for key in conv.keys():
            if key.startswith("session_"):
                turns = conv[key]
                if isinstance(turns, list) and turns:
                    print(f"  {key}: {len(turns)} turns")
                    first_turn = turns[0]
                    if isinstance(first_turn, dict):
                        print(f"    first turn keys: {list(first_turn.keys())}")
                        if 'dia_id' in first_turn:
                            print(f"    dia_id: {first_turn['dia_id']}")
                        if 'text' in first_turn:
                            text_snippet = first_turn['text'][:80]
                            print(f"    text: {text_snippet!r}...")
                    else:
                        print(f"    first turn type: {type(first_turn)}")

if __name__ == "__main__":
    main()
