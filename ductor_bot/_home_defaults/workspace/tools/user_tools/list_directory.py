#!/usr/bin/env python3
"""List files in a directory.

Usage:
    python tools/user_tools/list_directory.py --dir_path .
"""
import argparse
import os
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="List directory contents")
    parser.add_argument("--dir_path", default=".", help="Directory path")
    parser.add_argument("--path", help="Alias for dir_path")
    args = parser.parse_args()

    path = args.path if args.path else args.dir_path
    
    try:
        p = Path(path).resolve()
        if not p.is_dir():
            print(json.dumps({"error": f"Not a directory: {path}"}))
            return

        items = []
        for entry in os.scandir(p):
            items.append({
                "name": entry.name,
                "is_dir": entry.is_dir(),
                "size": entry.stat().st_size if entry.is_file() else None
            })
        
        print(json.dumps({"path": str(p), "items": items}, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()
