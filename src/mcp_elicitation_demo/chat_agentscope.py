"""AgentScope-based chat bridge: ReActAgent + MCP via AgentScope.

Drop-in replacement for `chat.py` with the same SSE event contract.
The agent loop (reasoning -> tool calls -> reply) is driven by
AgentScope's `Agent.reply_stream`; the MCP connection uses an
`MCPClient` subclass that injects an `elicitation_callback` into the
MCP 2.x `Client`, routing elicitation requests to the browser via
SSE and resuming when the user submits the form.

SSE events (identical to chat.py):
    {"type": "reasoning_delta", "text": str}
    {"type": "assistant_delta", "text": str}
    {"type": "tool_call", "name": str, "arguments": dict, "id": str}
    {"type": "tool_result", "id": str, "name": str, "result": str}
    {"type": "elicitation", "id": str, "message": str, "schema": dict}
    {"type": "error", "message": str}
    {"type": "done"}

LLM overrides (per chat session, provided by the browser):
    {"base_url": str|None, "api_key": str|None, "model": str|None}
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any

from mcp import Client as MCPSdkClient
from mcp import types as mcp_types

from agentscope.agent import Agent
from agentscope.mcp import MCPClient, HttpMCPConfig
from agentscope.message import Msg, TextBlock
from agentscope.permission import PermissionContext, PermissionMode
from agentscope.state import AgentState
from agentscope.model import OpenAIChatModel
from agentscope.credential import OpenAICredential
from agentscope.tool import Toolkit

from .settings import settings

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
    return SYSTEM_PROMPT_TMPL.format(
        today=datetime.now().strftime("%Y-%m-%d %A")
    )


def _mcp_url() -> str:
    return f"http://{settings.host}:{settings.port}{settings.path}"


class _BrowserElicitClient(MCPClient):
    """MCPClient that wires an async elicitation bridge into the
    MCP SDK client. The callback awaits a Future resolved by the
    browser's POST /api/elicitation/respond.

    mode='auto' (modern 2026-07-28): the server's tools use the
    SEP-2322 guard pattern - the SDK's input-required driver fires
    this callback for embedded ElicitRequests and retries the call
    with the answers, so no legacy back-channel is needed.
    """

    # Set per ChatSession; emits the SSE elicitation event and
    # returns the Future that the browser resolves.
    elicitation_bridge: Any = None

    def _create_sdk_client(self) -> MCPSdkClient:
        transport = self._create_http_client()

        async def on_elicit(
            context: Any, params: mcp_types.ElicitRequestParams
        ) -> mcp_types.ElicitResult:
            bridge = type(self).elicitation_bridge
            if bridge is None:
                return mcp_types.ElicitResult(action="decline", content=None)
            return await bridge(params)

        return MCPSdkClient(
            transport,
            mode="auto",
            elicitation_callback=on_elicit,
        )


class ChatSession:
    """One conversation backed by an AgentScope Agent + MCP client."""

    def __init__(self) -> None:
        self.pending: dict[str, asyncio.Future] = {}
        self._agent: Agent | None = None
        self._mcp_client: _BrowserElicitClient | None = None
        self._emit: Any = None
        self._llm_override: dict[str, Any] | None = None

    # -- Elicitation bridge ------------------------------------------------

    async def _on_elicitation(
        self, params: mcp_types.ElicitRequestParams
    ) -> mcp_types.ElicitResult:
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
        result = await fut
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

    # -- Agent assembly ------------------------------------------------------

    async def _ensure_agent(self, llm_override: dict[str, Any] | None) -> Agent:
        """Build (or rebuild on LLM override change) the AgentScope Agent."""
        ov = llm_override or {}
        cur = ov.get("model") or settings.llm_model
        if (
            self._agent is not None
            and self._llm_override == (llm_override or {})
        ):
            return self._agent

        await self.close()

        # MCP client with browser elicitation bridge.
        client = _BrowserElicitClient(
            name="elicitation_demo",
            is_stateful=True,
            mcp_config=HttpMCPConfig(url=_mcp_url()),
        )
        _BrowserElicitClient.elicitation_bridge = self._on_elicitation
        await client.connect()

        toolkit = Toolkit(mcps=[client])

        model = OpenAIChatModel(
            credential=OpenAICredential(
                api_key=ov.get("api_key") or settings.llm_api_key,
                base_url=ov.get("base_url") or settings.llm_base_url,
            ),
            model=cur,
            stream=True,
        )

        self._agent = Agent(
            name="assistant",
            system_prompt=_system_prompt(),
            model=model,
            toolkit=toolkit,
            # Trust the MCP tools here - the demo's own elicitation
            # flow handles human-in-the-loop already.
            state=AgentState(
                permission_context=PermissionContext(
                    mode=PermissionMode.BYPASS
                )
            ),
        )
        self._mcp_client = client
        self._llm_override = llm_override or {}
        return self._agent

    async def close(self) -> None:
        if self._mcp_client is not None:
            try:
                await self._mcp_client.close()
            except Exception:
                pass
        self._mcp_client = None
        self._agent = None
        self._llm_override = None

    # -- Main loop (streaming) -------------------------------------------------

    async def chat_stream(
        self,
        user_text: str,
        emit: Any,
        llm_override: dict[str, Any] | None = None,
    ) -> None:
        """Run one user turn through AgentScope; emit SSE events."""
        self._emit = emit
        try:
            agent = await self._ensure_agent(llm_override)
            await agent.observe(
                msgs=Msg(
                    name="user",
                    content=[TextBlock(type="text", text=user_text)],
                    role="user",
                )
            )

            tool_args: dict[str, dict] = {}  # tool_call_id -> parsed args
            tool_names: dict[str, str] = {}  # tool_call_id -> name
            tool_text: dict[str, list[str]] = {}  # tool_call_id -> text parts

            async for ev in agent.reply_stream():
                etype = getattr(ev, "type", None)
                etype = getattr(etype, "value", etype)
                if etype == "THINKING_BLOCK_DELTA":
                    emit({"type": "reasoning_delta", "text": ev.delta})
                elif etype == "TEXT_BLOCK_DELTA":
                    emit({"type": "assistant_delta", "text": ev.delta})
                elif etype == "TOOL_CALL_START":
                    tool_names[ev.tool_call_id] = ev.tool_call_name
                elif etype == "TOOL_CALL_DELTA":
                    tool_args.setdefault(ev.tool_call_id, {})
                    raw = getattr(ev, "delta", None)
                    if isinstance(raw, str):
                        # accumulate raw json fragments
                        tool_args[ev.tool_call_id].setdefault("_raw", "")
                        tool_args[ev.tool_call_id]["_raw"] += raw
                elif etype == "TOOL_CALL_END":
                    args = tool_args.get(ev.tool_call_id, {})
                    raw = args.get("_raw", "")
                    parsed: dict[str, Any] = {}
                    if raw:
                        import json as _json

                        try:
                            parsed = _json.loads(raw)
                        except ValueError:
                            parsed = {}
                    emit(
                        {
                            "type": "tool_call",
                            "id": ev.tool_call_id,
                            "name": tool_names.get(ev.tool_call_id, ""),
                            "arguments": parsed,
                        }
                    )
                elif etype == "TOOL_RESULT_TEXT_DELTA":
                    tool_text.setdefault(ev.tool_call_id, []).append(ev.delta)
                elif etype == "TOOL_RESULT_END":
                    emit(
                        {
                            "type": "tool_result",
                            "id": ev.tool_call_id,
                            "name": tool_names.get(ev.tool_call_id, ""),
                            "result": "".join(
                                tool_text.get(ev.tool_call_id, [])
                            ),
                        }
                    )

            emit({"type": "done"})
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
