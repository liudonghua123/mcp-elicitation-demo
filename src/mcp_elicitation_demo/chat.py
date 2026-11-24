"""Web chat bridge: OpenAI-compatible LLM (streamed) + in-process MCP client.

The LLM (default GLM) decides when to call MCP tools. Plain tools execute
directly; elicitation tools surface a form to the browser via SSE events,
and the browser's answer is injected back into the MCP elicitation handler.

SSE events sent to the browser as `data: {json}\\n\\n`:
    {"type": "reasoning_delta", "text": str}   - streamed thinking text
    {"type": "assistant_delta", "text": str}   - streamed answer text
    {"type": "tool_call", "name": str, "arguments": dict, "id": str}
    {"type": "tool_result", "id": str, "name": str, "result": str}
    {"type": "elicitation", "id": str, "message": str, "schema": dict,
     "tool": str}
    {"type": "error", "message": str}
    {"type": "done"}                           - turn complete

LLM overrides (per chat session, provided by the browser):
    {"base_url": str|None, "api_key": str|None, "model": str|None}
"""

from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import AsyncExitStack
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from mcp import types as mcp_types
from openai import OpenAI

from .settings import settings

MAX_TOOL_ROUNDS = 6
LLM_MAX_TOKENS = 2048

SYSTEM_PROMPT_TMPL = (
    "You are a helpful assistant backed by MCP tools. "
    "Use tools when they help answer the user. "
    "Some tools may pause to ask the user questions (elicitation); "
    "the user's answers arrive automatically. "
    "Today's date is {today}. "
    "Resolve relative dates ('tomorrow', 'next Friday') to concrete "
    "YYYY-MM-DD dates and times to HH:MM, and pass everything the user "
    "already provided as tool arguments so they are not asked again. "
    "Reply in the user's language. Be concise."
)


def _system_prompt() -> str:
    from datetime import datetime

    return SYSTEM_PROMPT_TMPL.format(today=datetime.now().strftime("%Y-%m-%d %A"))

# Tool schemas for the LLM, derived from the MCP server's tool list.
_llm_tools_cache: list[dict[str, Any]] | None = None


def _mcp_url() -> str:
    return f"http://{settings.host}:{settings.port}{settings.path}"


async def get_llm_tools() -> list[dict[str, Any]]:
    """List MCP tools and convert them to OpenAI function-calling schemas."""
    global _llm_tools_cache
    if _llm_tools_cache is not None:
        return _llm_tools_cache

    async with Client(StreamableHttpTransport(_mcp_url())) as client:
        tools = await client.list_tools()  # flat list in fastmcp

    llm_tools = []
    for t in tools:
        props = {}
        for name, spec in t.input_schema.get("properties", {}).items():
            props[name] = {k: v for k, v in spec.items() if k in ("type", "description", "enum")}
        llm_tools.append(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or t.name,
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": t.input_schema.get("required", []),
                    },
                },
            }
        )
    _llm_tools_cache = llm_tools
    return llm_tools


def _make_llm(llm_override: dict[str, Any] | None) -> OpenAI:
    """Build an OpenAI client, honoring per-session overrides."""
    ov = llm_override or {}
    return OpenAI(
        api_key=ov.get("api_key") or settings.llm_api_key,
        base_url=ov.get("base_url") or settings.llm_base_url,
    )


def _model_name(llm_override: dict[str, Any] | None) -> str:
    ov = llm_override or {}
    return ov.get("model") or settings.llm_model


class ChatSession:
    """One conversation: an MCP client connection with a browser-driven
    elicitation handler plus an OpenAI-compatible message history."""

    def __init__(self) -> None:
        self.history: list[dict[str, Any]] = []
        # Pending elicitations awaiting the browser: id -> asyncio.Future
        self.pending: dict[str, asyncio.Future] = {}
        self._mcp_client: Client | None = None
        self._stack: AsyncExitStack | None = None
        self._emit: Any = None  # active SSE emitter, set per request

    # -- MCP client lifecycle ------------------------------------------------

    async def _ensure_mcp(self) -> Client:
        if self._mcp_client is not None:
            return self._mcp_client
        self._stack = AsyncExitStack()

        async def elicitation_handler(
            message: str, response_type: Any, params: Any, context: Any
        ) -> mcp_types.ElicitResult:
            return await self._on_elicitation(params)

        client = Client(
            StreamableHttpTransport(_mcp_url()),
            elicitation_handler=elicitation_handler,
            # modern era (default 'auto'): the server uses the
            # SEP-2322 guard pattern, no legacy back-channel needed
        )
        await self._stack.enter_async_context(client)
        self._mcp_client = client
        return client

    async def close(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._mcp_client = None
        self._stack = None

    # -- Elicitation bridge --------------------------------------------------

    async def _on_elicitation(self, params: mcp_types.ElicitRequestParams) -> mcp_types.ElicitResult:
        if getattr(params, "mode", "form") != "form":
            return mcp_types.ElicitResult(action="decline", content=None)
        eid = uuid.uuid4().hex[:12]
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self.pending[eid] = fut
        self._emit(
            {
                "type": "elicitation",
                "id": eid,
                "message": params.message,
                "schema": params.requested_schema or {},
            }
        )
        result = await fut  # resolved by /api/elicitation/respond
        self.pending.pop(eid, None)
        return result

    def respond_elicitation(self, eid: str, action: str, content: Any) -> bool:
        fut = self.pending.get(eid)
        if fut is None or fut.done():
            return False
        if action == "accept":
            fut.set_result(mcp_types.ElicitResult(action="accept", content=content))
        elif action == "cancel":
            fut.set_result(mcp_types.ElicitResult(action="cancel", content=None))
        else:
            fut.set_result(mcp_types.ElicitResult(action="decline", content=None))
        return True

    # -- Tool execution -------------------------------------------------------

    async def _call_mcp_tool(self, name: str, arguments: dict) -> str:
        client = await self._ensure_mcp()
        res = await client.call_tool(name, arguments or {})
        parts = []
        for c in res.content:
            if hasattr(c, "text"):
                parts.append(c.text)
        return "\n".join(parts) or "(empty result)"

    # -- Main loop (streaming) --------------------------------------------------

    async def chat_stream(
        self,
        user_text: str,
        emit: Any,
        llm_override: dict[str, Any] | None = None,
    ) -> None:
        """Run one user turn with a fully streamed LLM; emit SSE events."""
        self._emit = emit
        llm = _make_llm(llm_override)
        model = _model_name(llm_override)
        llm_tools = await get_llm_tools()

        self.history.append({"role": "user", "content": user_text})
        messages = [{"role": "system", "content": _system_prompt()}, *self.history]

        def stream_once(msgs: list[dict]):
            """Blocking streamed completion; returns (text, reasoning, tool_calls)."""
            stream = llm.chat.completions.create(
                model=model,
                messages=msgs,
                tools=llm_tools,
                max_tokens=LLM_MAX_TOKENS,
                temperature=0.3,
                stream=True,
            )
            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            tool_calls: dict[int, dict] = {}
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                rc = getattr(delta, "reasoning_content", None)
                if rc:
                    reasoning_parts.append(rc)
                    emit({"type": "reasoning_delta", "text": rc})
                if delta.content:
                    text_parts.append(delta.content)
                    emit({"type": "assistant_delta", "text": delta.content})
                for tc in delta.tool_calls or []:
                    slot = tool_calls.setdefault(
                        tc.index, {"id": "", "name": "", "arguments": ""}
                    )
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            slot["name"] += tc.function.name
                        if tc.function.arguments:
                            slot["arguments"] += tc.function.arguments
            text = "".join(text_parts)
            reasoning = "".join(reasoning_parts)
            calls = [
                {"id": v["id"] or f"call_{k}", "name": v["name"], "arguments": v["arguments"]}
                for k, v in sorted(tool_calls.items())
            ]
            return text, reasoning, calls

        try:
            for _round in range(MAX_TOOL_ROUNDS):
                text, reasoning, calls = await asyncio.to_thread(stream_once, messages)

                if not calls:
                    self.history.append({"role": "assistant", "content": text})
                    emit({"type": "done"})
                    return

                # Record assistant turn with tool calls (wire format).
                assistant_entry: dict[str, Any] = {
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {"name": c["name"], "arguments": c["arguments"]},
                        }
                        for c in calls
                    ],
                }
                if reasoning:
                    assistant_entry["reasoning_content"] = reasoning
                self.history.append(assistant_entry)
                messages.append(assistant_entry)

                for c in calls:
                    try:
                        args = json.loads(c["arguments"] or "{}")
                    except ValueError:
                        args = {}
                    emit({"type": "tool_call", "id": c["id"], "name": c["name"], "arguments": args})
                    result = await self._call_mcp_tool(c["name"], args)
                    emit({"type": "tool_result", "id": c["id"], "name": c["name"], "result": result})
                    messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
                    self.history.append(
                        {"role": "tool", "tool_call_id": c["id"], "content": result}
                    )

            emit({"type": "error", "message": "Exceeded tool round limit."})
        except Exception as exc:
            emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})


_sessions: dict[str, ChatSession] = {}


def get_session(session_id: str) -> ChatSession:
    if session_id not in _sessions:
        _sessions[session_id] = ChatSession()
    return _sessions[session_id]


async def cleanup_session(session_id: str) -> None:
    sess = _sessions.pop(session_id, None)
    if sess is not None:
        await sess.close()
