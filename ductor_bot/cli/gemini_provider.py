"""Async wrapper around the Google Gemini CLI."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from collections.abc import AsyncGenerator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ductor_bot.cli.base import BaseCLI, CLIConfig, docker_wrap
from ductor_bot.cli.stream_events import (
    ResultEvent,
    StreamEvent,
    SystemInitEvent,
    ToolUseEvent,
    parse_stream_line,
)
from ductor_bot.cli.types import CLIResponse

logger = logging.getLogger(__name__)


class GeminiCLI(BaseCLI):
    """Async wrapper around the Google Gemini CLI."""

    def __init__(self, config: CLIConfig) -> None:
        self._config = config
        self._working_dir = Path(config.working_dir).resolve()
        self._cli_js = self._find_cli_js()
        logger.info("Gemini CLI wrapper: cwd=%s, model=%s", self._working_dir, config.model)

    @staticmethod
    def _find_cli_js() -> str | None:
        """Find the absolute path to the Gemini CLI's index.js to bypass wrappers."""
        if os.name != "nt":
            return None

        prefixes = [
            Path("C:/nvm4w/nodejs"),
            Path(os.environ.get("APPDATA", "")) / "npm",
            Path(os.environ.get("ProgramFiles", "")) / "nodejs",
        ]

        for p in prefixes:
            js = p / "node_modules" / "@google" / "gemini-cli" / "dist" / "index.js"
            if js.is_file():
                return str(js)

        return None

    def _build_command(
        self,
        resume_session: str | None = None,
        continue_session: bool = False,
        streaming: bool = False,
    ) -> list[str]:
        cfg = self._config
        cmd = ["node", self._cli_js] if self._cli_js else ["gemini"]

        # Real prompt is piped via stdin to avoid WinError 206 (command line too long).
        # Gemini CLI headless mode correctly handles stdin as the primary prompt.
        cmd += ["--output-format", "stream-json" if streaming else "json"]

        # Ensure the current workspace is included to avoid "Safe Mode" tool restrictions
        cmd += ["--include-directories", "."]

        if cfg.model:
            cmd += ["--model", cfg.model]
        if cfg.permission_mode == "bypassPermissions":
            cmd += ["--approval-mode", "yolo"]
        if resume_session:
            cmd += ["--resume", resume_session]
        elif continue_session:
            cmd += ["--resume", "latest"]
        if cfg.allowed_tools:
            cmd += ["--allowed-tools", *cfg.allowed_tools]
        if cfg.cli_parameters:
            cmd.extend(cfg.cli_parameters)

        return cmd

    def _prepare_env(self, system_prompt_path: str | None = None) -> dict[str, str]:
        """Prepare environment variables for the Gemini CLI subprocess."""
        env = os.environ.copy()
        env["ANTIGRAVITY_IDE_ENABLED"] = "false"
        env["GEMINI_IDE_ENABLED"] = "false"
        if system_prompt_path:
            env["GEMINI_SYSTEM_MD"] = system_prompt_path
        return env

    @contextmanager
    def _system_prompt_file(self) -> Iterator[str | None]:
        """Managed temporary file for the system prompt with Windows safety rules."""
        sys_p = self._config.system_prompt or ""
        app_p = self._config.append_system_prompt or ""
        
        # Mandatory instructions to avoid interactive hangs on Windows.
        safety_p = (
            "\n\n## Windows Safety Rules\n"
            "- When using run_shell_command, NEVER run interactive commands.\n"
            "- On Windows, `curl` is an alias for `Invoke-WebRequest`. For the real tool, use `curl.exe`.\n"
            "- To avoid hangs, always use `-UseBasicParsing` or `-s` (silent) with web requests.\n"
            "- Redirect stdin to null for shell commands: `cmd ... < $null` in PowerShell."
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as tf:
            path = tf.name
            tf.write(sys_p)
            if app_p:
                tf.write("\n\n" + app_p)
            tf.write(safety_p)

        try:
            yield path
        finally:
            if os.path.exists(path):
                os.unlink(path)

    async def send(
        self,
        prompt: str,
        resume_session: str | None = None,
        continue_session: bool = False,
        timeout_seconds: float | None = None,
    ) -> CLIResponse:
        """Send a prompt and return final result, leveraging streaming logic for tools."""
        last_event = None
        full_text = []
        session_id = resume_session
        usage = {}

        async for event in self.send_streaming(prompt, resume_session, continue_session, timeout_seconds):
            if event.type == "assistant" and hasattr(event, "text"):
                full_text.append(event.text)
            if isinstance(event, ResultEvent):
                last_event = event
                session_id = event.session_id or session_id
                usage = event.usage

        if not last_event and not full_text:
            return CLIResponse(result="", is_error=True)

        return CLIResponse(
            session_id=session_id,
            result="".join(full_text),
            is_error=getattr(last_event, "is_error", False) if last_event else False,
            usage=usage
        )

    async def send_streaming(
        self,
        prompt: str,
        resume_session: str | None = None,
        continue_session: bool = False,
        timeout_seconds: float | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Send a prompt and yield stream events, handling recursive tools and history filtering."""
        current_prompt = prompt
        current_resume = resume_session
        max_turns = self._config.max_turns or 10
        turn = 0
        state = _StreamState(session_id=resume_session)

        with self._system_prompt_file() as sys_path:
            env = self._prepare_env(sys_path)
            while turn < max_turns:
                turn += 1
                cmd = self._build_command(
                    resume_session=current_resume, 
                    continue_session=continue_session, 
                    streaming=True
                )
                _log_cmd(cmd, streaming=True)

                exec_cmd, use_cwd = docker_wrap(
                    cmd, self._config.docker_container, self._config.chat_id, self._working_dir
                )
                
                # Clear pending tools for the new turn
                state.pending_tools = []
                async for event in self._run_streaming_turn(exec_cmd, use_cwd, env, state, current_prompt, timeout_seconds):
                    yield event

                if state.error_occurred or not state.pending_tools:
                    break

                # Execute remaining tools (e.g. ask_user or failed internal tools)
                results = []
                for call in state.pending_tools:
                    out = await self._execute_tool(call["name"], call["arguments"])
                    results.append({"call_id": call["call_id"], "tool_name": call["name"], "output": out})

                current_prompt = self._format_tool_results(results)
                current_resume = state.session_id
                continue_session = False

            if state.session_id:
                yield ResultEvent(type="result", session_id=state.session_id, result="", usage=state.usage)

    async def _run_streaming_turn(
        self,
        cmd: list[str],
        cwd: Path,
        env: dict[str, str],
        state: _StreamState,
        current_prompt: str,
        timeout_seconds: float | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Executes a single CLI process turn and yields filtered events."""
        stdin_val = current_prompt.encode()
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            limit=4 * 1024 * 1024,
            env=env,
        )
        if not process.stdout or not process.stderr or not process.stdin:
            raise RuntimeError("Subprocess created without pipes")

        process.stdin.write(stdin_val)
        await process.stdin.drain()
        process.stdin.close()

        reg = self._config.process_registry
        tracked = reg.register(self._config.chat_id, process, self._config.process_label) if reg else None
        start_t = time.time()
        seen_new = False
        stderr_task = asyncio.create_task(process.stderr.read())

        # Map standard tool names to ductor's script names
        tool_name_map = {
            "read_file": "read_document",
            "list_directory": "list_files",
            "list_files": "list_files",
            "google_web_search": "google_web_search",
            "google_search": "google_web_search",
            "web_fetch": "web_fetch",
            "run_shell_command": "run_shell_command",
            "ask_user": "ask_user"
        }

        turn_requested_tools: dict[str, dict[str, Any]] = {}
        turn_completed_tool_ids: set[str] = set()
        
        # Tools we ALWAYS want to run manually to ensure Windows optimizations
        OVERRIDE_TOOLS = {"ask_user", "run_shell_command", "google_web_search", "google_search", "transcribe_audio"}

        try:
            async with asyncio.timeout(timeout_seconds or 300.0):
                while True:
                    line_b = await process.stdout.readline()
                    if not line_b:
                        break
                    line = line_b.decode(errors="replace").rstrip()
                    if not line:
                        continue

                    logger.info("Gemini raw line: %s", line)
                    if not line.startswith("{"):
                        if not seen_new:
                            yield StreamEvent(type="message", content=line + "\n", role="assistant")
                        continue

                    for event in parse_stream_line(line, last_session_id=state.session_id):
                        if getattr(event, "session_id", None):
                            state.session_id = event.session_id

                        if event.type in ("system", "init"):
                            continue
                        
                        if event.delta or isinstance(event, ToolUseEvent) or (event.type == "assistant" and getattr(event, "text", "")):
                            seen_new = True
                        
                        if not seen_new:
                            continue

                        if isinstance(event, ToolUseEvent):
                            turn_requested_tools[event.call_id] = {
                                "name": event.tool_name,
                                "arguments": event.arguments,
                                "call_id": event.call_id
                            }

                        if event.type == "tool_result":
                            tid = getattr(event, "tool_id", None)
                            if tid:
                                req = turn_requested_tools.get(tid, {})
                                # Only mark as completed if it succeeded internally AND isn't an override tool
                                if getattr(event, "status", "") == "success" and req.get("name") not in OVERRIDE_TOOLS:
                                    turn_completed_tool_ids.add(tid)
                            continue

                        if isinstance(event, ResultEvent):
                            state.add_usage(event.usage)
                            if event.is_error and not seen_new:
                                state.error_occurred = True
                                yield event
                                return
                            continue

                        yield event

        except TimeoutError:
            process.kill()
            yield ResultEvent(type="result", result="Timeout", is_error=True, session_id=state.session_id)
            state.error_occurred = True
        finally:
            if tracked and reg:
                reg.unregister(tracked)
            stderr_bytes = await stderr_task
            await process.wait()
            logger.info("CLI turn finished in %.2fs (exit=%d)", time.time() - start_t, process.returncode)
            
            # Identify tools that need manual fallback or override
            manual_tools = [t for tid, t in turn_requested_tools.items() if tid not in turn_completed_tool_ids]
            for t in manual_tools:
                orig_name = t["name"]
                t["name"] = tool_name_map.get(orig_name, orig_name)
                state.pending_tools.append(t)

            if process.returncode != 0 and not seen_new:
                state.error_occurred = True
                yield ResultEvent(type="result", result=stderr_bytes.decode(errors="replace")[:500], is_error=True, session_id=state.session_id)

    def _format_tool_results(self, results: list[dict[str, Any]]) -> str:
        return json.dumps({
            "role": "assistant",
            "content": [{"type": "tool_result", "tool_id": r["call_id"], "output": r["output"]} for r in results]
        })

    async def _execute_tool(self, name: str, arguments: dict[str, Any]) -> str:
        tool_dirs = ["telegram_tools", "user_tools", "cron_tools", "webhook_tools"]
        tool_path = next((p for d in tool_dirs if (p := self._working_dir / "tools" / d / f"{name}.py").exists()), None)
        if not tool_path:
            return f"Error: Tool {name} not found"

        cmd = [sys.executable, str(tool_path)]
        for k, v in arguments.items():
            cmd += [f"--{k}", str(v)]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=self._working_dir
            )
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60.0)
                return stdout.decode(errors='replace') if process.returncode == 0 else f"Error: {stderr.decode()}"
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return "Error: Tool execution timed out after 60s"
        except Exception as e:
            return f"Exception: {e}"


class _StreamState:
    def __init__(self, session_id: str | None = None):
        self.session_id = session_id
        self.usage = {"input_tokens": 0, "output_tokens": 0}
        self.error_occurred = False
        self.pending_tools: list[dict[str, Any]] = []

    def add_usage(self, usage: dict[str, Any] | None) -> None:
        if usage:
            self.usage["input_tokens"] += usage.get("input_tokens", 0)
            self.usage["output_tokens"] += usage.get("output_tokens", 0)


def _log_cmd(cmd: list[str], *, streaming: bool = False) -> None:
    safe = [(c[:80] + "...") if len(c) > 80 and i > 0 and cmd[i - 1].startswith("--") else c for i, c in enumerate(cmd)]
    logger.info("%s: %s", "CLI stream cmd" if streaming else "CLI cmd", " ".join(safe))


def _parse_response(stdout: bytes, stderr: bytes, returncode: int | None) -> CLIResponse:
    stderr_text = stderr.decode(errors="replace")[:2000] if stderr else ""
    raw = stdout.decode().strip()
    if not raw:
        return CLIResponse(result="", is_error=True, returncode=returncode, stderr=stderr_text)
    try:
        data = json.loads(raw)
        res = data.get("response") or data.get("content") or data.get("output") or raw
        sid, stats = data.get("session_id"), data.get("stats", {})
        usage = {"input_tokens": stats.get("input_tokens", 0), "output_tokens": stats.get("output_tokens", 0)}
    except json.JSONDecodeError:
        res, sid, usage = raw, None, {}
    return CLIResponse(session_id=sid, result=res, is_error=returncode != 0, returncode=returncode, stderr=stderr_text, usage=usage)
