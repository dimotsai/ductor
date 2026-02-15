#!/usr/bin/env python3
"""Ask the user a question with optional buttons.

Usage:
    python tools/user_tools/ask_user.py --question "Do you like cats?" --type yesno
    python tools/user_tools/ask_user.py --questions '[{"question": "...", "type": "choice", "options": [...]}]'
"""
from __future__ import annotations

import argparse
import json
import sys
import ast

def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the user a question")
    parser.add_argument("--questions", help="JSON string of multiple questions")
    parser.add_argument("--question", help="The question to ask")
    parser.add_argument("--type", choices=["text", "choice", "yesno"], default="text", help="Question type")
    parser.add_argument("--options", help="JSON string of options (for choice type)")
    parser.add_argument("--header", help="Short label for the question")
    
    args = parser.parse_args()

    # If questions list is provided, take the first one
    if args.questions:
        try:
            # Try standard JSON first
            try:
                qs = json.loads(args.questions)
            except json.JSONDecodeError:
                # Fallback: shell might have converted double quotes to single quotes
                # Use ast.literal_eval for Python-like string representation
                qs = ast.literal_eval(args.questions)
            
            if isinstance(qs, list) and len(qs) > 0:
                q = qs[0]
                args.question = q.get("question")
                args.type = q.get("type", "text")
                opts = q.get("options", [])
                args.options = json.dumps(opts) if isinstance(opts, (list, dict)) else str(opts)
                args.header = q.get("header")
        except Exception as e:
            sys.stderr.write(f"Error parsing questions data: {e}\n")

    if not args.question:
        sys.stderr.write("Error: No question provided.\n")
        sys.exit(1)

    output = [args.question]
    
    if args.type == "yesno":
        output.append("\n[button:Yes] [button:No]")
    elif args.type == "choice" and args.options:
        try:
            try:
                options = json.loads(args.options)
            except json.JSONDecodeError:
                options = ast.literal_eval(args.options)
                
            btns = []
            if isinstance(options, list):
                for opt in options:
                    if isinstance(opt, dict):
                        label = opt.get("label") or opt.get("text")
                    else:
                        label = str(opt)
                    if label:
                        btns.append(f"[button:{label}]")
            if btns:
                output.append("\n" + "\n".join(btns))
        except Exception:
            pass
            
    print("\n".join(output))

if __name__ == "__main__":
    main()
