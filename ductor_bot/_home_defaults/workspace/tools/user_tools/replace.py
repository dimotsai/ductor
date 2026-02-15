#!/usr/bin/env python3
"""Replace text in a file.

Usage:
    python tools/user_tools/replace.py --file_path path/to/file --old_string "old" --new_string "new"
"""
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Replace text in a file")
    parser.add_argument("--file_path", required=True, help="Path to the file")
    parser.add_argument("--path", help="Alias for file_path")
    parser.add_argument("--old_string", required=True, help="Text to replace")
    parser.add_argument("--new_string", required=True, help="New text")
    parser.add_argument("--expected_replacements", type=int, help="Expected number of replacements")
    parser.add_argument("--instruction", help="Instruction describing the change")
    args = parser.parse_args()

    path = args.path if args.path else args.file_path
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if args.old_string not in content:
            print(f"Error: old_string not found in {path}")
            sys.exit(1)
            
        count = content.count(args.old_string)
        if args.expected_replacements is not None and count != args.expected_replacements:
            print(f"Error: Expected {args.expected_replacements} replacements, but found {count}")
            sys.exit(1)
            
        new_content = content.replace(args.old_string, args.new_string)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Successfully replaced text in {path} ({count} occurrence(s))")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
