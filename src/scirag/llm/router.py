"""Single entry point for all LLM calls. Open-source (Ollama) and frontier
(Anthropic/OpenAI) models live behind the same `complete()` call; which one a
given agent uses is decided in configs/models.yaml.
"""

from __future__ import annotations

import litellm

from scirag.config import backend_for

# Let LiteLLM drop params a given provider doesn't support instead of erroring.
litellm.drop_params = True


def _complete_claude_cli(messages: list[dict[str, str]]) -> str:
    """Call `claude -p` as a subprocess using the Claude Code CLI (Plus subscription)."""
    import shutil
    import subprocess

    if not shutil.which("claude"):
        raise RuntimeError(
            "`claude` CLI not found in PATH. Install Claude Code: https://claude.ai/code"
        )

    # Flatten messages into a single prompt: system preamble + turns
    parts = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "system":
            parts.append(content)
        else:
            parts.append(f"{role.upper()}: {content}")
    prompt = "\n\n".join(parts)

    result = subprocess.run(
        ["claude", "-p", prompt],
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


def complete(
    agent: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    """Run a chat completion for the model bound to `agent` in models.yaml."""
    backend = backend_for(agent)

    if backend["model"] == "claude-code":
        return _complete_claude_cli(messages)

    kwargs: dict = {
        "model": backend["model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
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
    max_tokens: int = 1200,
):
    """Async streaming completion — yields text tokens as they arrive."""
    backend = backend_for(agent)

    if backend["model"] == "claude-code":
        # claude -p doesn't stream; yield the full response as one chunk
        yield _complete_claude_cli(messages)
        return

    kwargs: dict = {
        "model": backend["model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if "api_base" in backend:
        kwargs["api_base"] = backend["api_base"]
    response = await litellm.acompletion(**kwargs)
    async for chunk in response:
        token = chunk["choices"][0]["delta"].get("content") or ""
        if token:
            yield token
