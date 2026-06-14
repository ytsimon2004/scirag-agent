"""Single entry point for all LLM calls. Open-source (Ollama) and frontier
(Anthropic/OpenAI) models live behind the same `complete()` call; which one a
given agent uses is decided in configs/models.yaml.
"""

from __future__ import annotations

import litellm

from scirag.config import backend_for, get_effort

# Let LiteLLM drop params a given provider doesn't support instead of erroring.
litellm.drop_params = True

# Ollama models that support hybrid thinking (toggled via the `think` option).
_THINKING_HINTS = ("qwen3", "deepseek-r1", "magistral")

# Token budget per effort level — gives a real gradient for thinking models, where
# medium/high both keep thinking on and only differ in how long the model may run.
_EFFORT_MAX_TOKENS = {"low": 800, "medium": 1200, "high": 2048}


def _reasoning_kwargs(backend: dict, effort: str) -> dict:
    """Map an effort level to provider-specific reasoning controls.

    Ollama thinking models toggle the `think` option directly (litellm maps
    `reasoning_effort` to `think: True` for *every* level, so it can't distinguish
    them). Frontier backends use litellm's unified `reasoning_effort`, dropped
    automatically by `drop_params` where unsupported.
    """
    model = backend["model"]
    if "api_base" in backend:  # local/Ollama backends
        if any(h in model for h in _THINKING_HINTS):
            return {"think": effort != "low"}  # low = thinking off (fast)
        return {}  # non-thinking ollama model: sending `think` would error
    return {"reasoning_effort": effort}


def _flatten_messages(messages: list[dict[str, str]]) -> str:
    """Flatten chat messages into a single prompt string for CLI backends."""
    parts = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "system":
            parts.append(content)
        else:
            parts.append(f"{role.upper()}: {content}")
    return "\n\n".join(parts)


def _complete_claude_cli(messages: list[dict[str, str]], effort: str = "medium") -> str:
    """Call `claude -p` as a subprocess using the Claude Code CLI (Plus subscription)."""
    import shutil
    import subprocess

    if not shutil.which("claude"):
        raise RuntimeError(
            "`claude` CLI not found in PATH. Install Claude Code: https://claude.ai/code"
        )

    result = subprocess.run(
        ["claude", "-p", "--effort", effort, _flatten_messages(messages)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if any(w in stderr.lower() for w in ("auth", "login", "unauthorized", "not logged")):
            raise RuntimeError(
                "Claude CLI is not authenticated. Run `claude` to log in, then retry."
            )
        raise RuntimeError(f"`claude -p` failed: {stderr}")
    return result.stdout.strip()


def _complete_codex_cli(messages: list[dict[str, str]], effort: str = "medium") -> str:
    """Call `codex exec` as a subprocess using the OpenAI Codex CLI (OpenAI subscription)."""
    import shutil
    import subprocess
    import tempfile

    if not shutil.which("codex"):
        raise RuntimeError(
            "`codex` CLI not found in PATH. Install it: https://github.com/openai/codex"
        )

    with tempfile.NamedTemporaryFile(mode="r", suffix=".txt", delete=False) as tmp:
        tmp_path = tmp.name

    result = subprocess.run(
        [
            "codex",
            "exec",
            "-c",
            f"model_reasoning_effort={effort}",
            "--output-last-message",
            tmp_path,
            _flatten_messages(messages),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if any(w in stderr.lower() for w in ("auth", "login", "unauthorized", "not logged")):
            raise RuntimeError("Codex CLI is not authenticated. Run `codex login`, then retry.")
        raise RuntimeError(f"`codex exec` failed: {stderr}")

    import os

    answer = open(tmp_path).read().strip()
    os.unlink(tmp_path)
    return answer


def complete(
    agent: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    effort: str | None = None,
) -> str:
    """Run a chat completion for the model bound to `agent` in models.yaml."""
    backend = backend_for(agent)
    effort = effort or get_effort()
    if max_tokens is None:
        max_tokens = _EFFORT_MAX_TOKENS[effort]

    if backend["model"] == "claude-code":
        return _complete_claude_cli(messages, effort)
    if backend["model"] == "codex":
        return _complete_codex_cli(messages, effort)

    kwargs: dict = {
        "model": backend["model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        **_reasoning_kwargs(backend, effort),
    }
    if "api_base" in backend:  # local/Ollama backends
        kwargs["api_base"] = backend["api_base"]
    resp = litellm.completion(**kwargs)
    return resp["choices"][0]["message"]["content"]


async def complete_stream(
    agent: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    effort: str | None = None,
):
    """Async streaming completion — yields text tokens as they arrive."""
    backend = backend_for(agent)
    effort = effort or get_effort()
    if max_tokens is None:
        max_tokens = _EFFORT_MAX_TOKENS[effort]

    if backend["model"] == "claude-code":
        yield _complete_claude_cli(messages, effort)
        return
    if backend["model"] == "codex":
        yield _complete_codex_cli(messages, effort)
        return

    kwargs: dict = {
        "model": backend["model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        **_reasoning_kwargs(backend, effort),
    }
    if "api_base" in backend:
        kwargs["api_base"] = backend["api_base"]
    response = await litellm.acompletion(**kwargs)
    async for chunk in response:
        token = chunk["choices"][0]["delta"].get("content") or ""
        if token:
            yield token
