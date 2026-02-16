"""Async wrapper around the Google Gemini CLI (Aligned with Claude-style logic)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from collections.abc import AsyncGenerator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ductor_bot.cli.base import BaseCLI, CLIConfig, docker_wrap
from ductor_bot.cli.stream_events import (
    AssistantTextDelta,
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
        logger.info("Gemini CLI wrapper (Claude-style): cwd=%s, model=%s", self._working_dir, config.model)

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

        # Aligned with Claude: stream-json for real-time events
        cmd += ["--output-format", "stream-json" if streaming else "json"]
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
        env = os.environ.copy()
        env["ANTIGRAVITY_IDE_ENABLED"] = "false"
        env["ANTIGRAVITY_IDE_DANGEROUS_YOLO_MODE"] = "true"
        env["GEMINI_IDE_ENABLED"] = "false"
        if system_prompt_path:
            env["GEMINI_SYSTEM_MD"] = system_prompt_path
        return env

    @contextmanager
    def _system_prompt_file(self) -> Iterator[str | None]:
        sys_p = self._config.system_prompt or ""
        app_p = self._config.append_system_prompt or ""
        safety_p = (
            "\n\n## Windows Safety Rules\n"
            "- When using run_shell_command, NEVER run interactive commands.\n"
            "- Redirect stdin to null for shell commands: `cmd ... < $null` in PowerShell."
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as tf:
            path = tf.name
            tf.write(sys_p + "\n\n" + app_p + safety_p)

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
        """Simplified send (non-streaming)."""
        cmd = self._build_command(resume_session, continue_session, streaming=False)
        exec_cmd, use_cwd = docker_wrap(cmd, self._config.docker_container, self._config.chat_id, self._working_dir)
        
        process = await asyncio.create_subprocess_exec(
            *exec_cmd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=use_cwd
        )
        stdout, stderr = await process.communicate(input=prompt.encode())
        return _parse_response(stdout, stderr, process.returncode)

    async def send_streaming(
        self,
        prompt: str,
        resume_session: str | None = None,
        continue_session: bool = False,
        timeout_seconds: float | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Claude-style streaming: Single process, no recursive loops."""
        cmd = self._build_command(resume_session, continue_session, streaming=True)
        _log_cmd(cmd, streaming=True)

        exec_cmd, use_cwd = docker_wrap(cmd, self._config.docker_container, self._config.chat_id, self._working_dir)
        
        with self._system_prompt_file() as sys_path:
            env = self._prepare_env(sys_path)
            process = await asyncio.create_subprocess_exec(
                *exec_cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=use_cwd,
                env=env,
            )
            
            # Start stderr drain in background
            stderr_task = asyncio.create_task(process.stderr.read())
            
            # Feed prompt and close stdin
            if process.stdin:
                process.stdin.write(prompt.encode())
                await process.stdin.drain()
                process.stdin.close()

            seen_new = False
            last_session_id = resume_session

            try:
                async with asyncio.timeout(timeout_seconds or 300.0):
                    while True:
                        line_b = await process.stdout.readline()
                        if not line_b:
                            break
                        line = line_b.decode(errors="replace").rstrip()
                        if not line:
                            continue

                        for event in parse_stream_line(line, last_session_id=last_session_id):
                            if getattr(event, "session_id", None):
                                last_session_id = event.session_id

                            # Progress tracking logic (like Claude)
                            if event.type in ("system", "init"):
                                continue

                            # Result events are the final word
                            if isinstance(event, ResultEvent):
                                yield event
                                return

                            # Gemini-specific: skip re-echoed history
                            if event.delta or isinstance(event, ToolUseEvent) or (event.type == "assistant" and getattr(event, "text", "")):
                                seen_new = True

                            if seen_new:
                                yield event

            except TimeoutError:
                process.kill()
                yield ResultEvent(type="result", result="Timeout", is_error=True, session_id=last_session_id)
            finally:
                await process.wait()
                await stderr_task

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
