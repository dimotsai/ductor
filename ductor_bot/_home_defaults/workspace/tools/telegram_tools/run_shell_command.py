"""Execute a shell command on the host system with Windows-specific optimizations."""

import argparse
import subprocess
import sys
import os

def run_command(command: str, timeout: int = 60):
    """Run a shell command and return its output."""
    try:
        # On Windows, we use powershell for better compatibility
        if os.name == "nt":
            # 1. Disable progress bar (speeds up Invoke-WebRequest/curl)
            # 2. Redirect stdin to null to avoid interactive hangs
            # 3. Use NoProfile to avoid user script interference
            ps_command = f"$ProgressPreference = 'SilentlyContinue'; {command}"
            process = subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command", ps_command],
                capture_output=True,
                text=True,
                timeout=timeout,
                stdin=subprocess.DEVNULL
            )
        else:
            process = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                executable="/bin/bash",
                stdin=subprocess.DEVNULL
            )
        
        output = process.stdout
        if process.stderr:
            output += f"\nError Output:\n{process.stderr}"
            
        if process.returncode != 0:
            output += f"\nProcess exited with code {process.returncode}"
            
        return output
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout} seconds."
    except Exception as e:
        return f"Error executing command: {str(e)}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a shell command.")
    parser.add_argument("command", nargs='?', help="The command to run")
    # Some agents pass command as a named argument
    parser.add_argument("--command", dest="cmd_named", help="The command to run (named)")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout in seconds")
    
    args = parser.parse_args()
    cmd = args.cmd_named if args.cmd_named else args.command
    
    if not cmd:
        # Handle cases where agent might pass the command as a single string positional
        print("Error: No command provided")
        sys.exit(1)
        
    print(run_command(cmd, args.timeout))
