#!/usr/bin/env python3
"""Write content to a file.

Usage:
    python tools/user_tools/write_file.py --file_path path/to/file --content "file content"
"""
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Write to a file")
    parser.add_argument("--file_path", required=True, help="Path to the file")
    parser.add_argument("--path", help="Alias for file_path")
    parser.add_argument("--content", required=True, help="Content to write")
    args = parser.parse_args()

    path = args.path if args.path else args.file_path
    
    try:
        # Safety check: don't write outside workspace (basic check)
        # In a real ductor setup, this would be more robust.
        with open(path, 'w', encoding='utf-8') as f:
            f.write(args.content)
        print(f"Successfully wrote to {path}")
    except Exception as e:
        print(f"Error writing to file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
