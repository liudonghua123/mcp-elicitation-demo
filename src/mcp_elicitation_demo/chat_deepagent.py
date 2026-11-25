"""DeepAgents / LangGraph chat bridge with native MCP elicitation interrupts.

This is the `deepagent` branch counterpart of `chat.py`. The agent loop is
owned by DeepAgents (built on LangGraph); MCP tools are discovered through
``langchain.mcp.MCPAdapter``. Modern LangChain adapters translate a server's
``InputRequiredResult`` into a LangGraph ``interrupt()`` carrying an
``MCPElicitationInterrupt`` payload. We surface that payload to the browser as
an SSE ``elicitation`` event, then resume the very same graph run with
``Command(resume={"responses": ...})`` once the user submits the form.

SSE events (identical to chat.py / chat_agentscope.py):
    {"type": "reasoning_delta", "text": str}
    {"type": "assistant_delta", "text": str}
    {"type": "tool_call", "name": str, "arguments": dict, "id": str}
    {"type": "tool_result", "id": str, "name": str, "result": str}
    {"type": "elicitation", "id": str, "message": str, "schema": dict,
     "tool": str}
    {"type": "error", "message": str}
    {"type": "done"}

LLM overrides (per chat session, provided by the browser):
    {"base_url": str|None, "api_key": str|None, "model": str|None}
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from typing import Any

from deepagents import create_deep_agent
from langchain.mcp import MCPAdapter
from langchain.mcp.elicitation import ELICITATION_INTERRUPT_TYPE
from langchain_core.messages import AIMessageChunk, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from .settings import settings

SYSTEM_PROMPT_TMPL = (
    "You are a helpful assistant backed by MCP tools. "
    "Use the MCP tools to answer; do not use file-system or planning tools "
    "for this demo. Some tools may pause to ask the user questions "
    "(elicitation); the user's answers arrive automatically. "
    "Today's date is {today}. "
    "Resolve relative dates ('tomorrow', 'next Friday') to concrete "
    "YYYY-MM-DD dates and times to HH:MM, and pass everything the user "
    "already provided as tool arguments so they are not asked again. "
    "Reply in the user's language. Be concise."
)

MAX_ELICITATION_ROUNDS = 12


def _system_prompt() -> str:
    return SYSTEM_PROMPT_TMPL.format(today=datetime.now().strftime("%Y-%m-%d %A"))


def _mcp_url() -> str:
    return f"http://{settings.host}:{settings.port}{settings.path}"


def _text_of(content: Any) -> str:
    """Normalize a LangChain v1 message content (str or block list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
            elif hasattr(block, "text"):
                parts.append(str(block.text))
        return "".join(parts)
    return str(content or "")


def _extract_interrupts(data: Any) -> list[dict[str, Any]]:
    """Find MCP elicitation payloads in a LangGraph `updates` chunk.

    Legacy astream yields `{"__interrupt__": (Interrupt(value=...), ...)}`;
    some versions pass a bare tuple/list of Interrupt objects instead.
    """
    found: list[dict[str, Any]] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "__interrupt__":
                    for item in value if isinstance(value, (list, tuple)) else [value]:
                        payload = getattr(item, "value", item)
                        if isinstance(payload, dict):
                            found.append(payload)
                else:
                    visit(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                visit(item)

    visit(data)
    return [
        p for p in found if p.get("type") == ELICITATION_INTERRUPT_TYPE
    ]


class ChatSession:
    """One conversation: a DeepAgents graph with a per-session checkpointer."""

    def __init__(self) -> None:
        # Browser answers to a pending form: eid -> Future[(action, content)]
        self.pending: dict[str, asyncio.Future] = {}
        self._agent: Any = None
        self._checkpointer = InMemorySaver()
        self._thread_id = uuid.uuid4().hex
        self._llm_key: tuple[Any, ...] | None = None
        self._emit: Any = None

    # -- Elicitation bridge ------------------------------------------------

    def respond_elicitation(self, eid: str, action: str, content: Any) -> bool:
        fut = self.pending.get(eid)
        if fut is None or fut.done():
            return False
        fut.set_result((action, content))
        return True

    async def _ask_browser(
        self, payload: dict[str, Any], emit: Any
    ) -> dict[str, Any]:
        """Emit one SSE form per request, wait for all browser answers."""
        waits: list[tuple[str, str, asyncio.Future]] = []
        responses: dict[str, Any] = {}

        for req in payload.get("requests", []):
            key = req["key"]
            if req.get("mode") != "form":
                # Only form elicitation is supported by this demo UI.
                responses[key] = {"action": "decline"}
                continue
            eid = uuid.uuid4().hex[:12]
            fut: asyncio.Future = asyncio.get_running_loop().create_future()
            self.pending[eid] = fut
            waits.append((key, eid, fut))
            emit(
                {
                    "type": "elicitation",
                    "id": eid,
                    "message": req.get("message", ""),
                    "schema": req.get("requested_schema") or {},
                    "tool": payload.get("tool_name", ""),
                }
            )

        results = await asyncio.gather(*(f for _, _, f in waits), return_exceptions=True)
        for (key, eid, _fut), res in zip(waits, results):
            self.pending.pop(eid, None)
            action, content = (
                res if isinstance(res, tuple) and len(res) == 2 else ("decline", None)
            )
            if action == "accept":
                responses[key] = {"action": "accept", "content": content}
            elif action == "cancel":
                responses[key] = {"action": "cancel"}
            else:
                responses[key] = {"action": "decline"}

        return {"responses": responses}

    # -- Agent assembly ----------------------------------------------------

    async def _ensure_agent(self, llm_override: dict[str, Any] | None) -> Any:
        ov = llm_override or {}
        key = (ov.get("api_key"), ov.get("base_url"), ov.get("model"))
        if self._agent is not None and self._llm_key == key:
            return self._agent

        # The adapted tools keep the client alive after the context exits.
        async with MCPAdapter(_mcp_url()) as adapter:
            tools = await adapter.list_tools()

        model = ChatOpenAI(
            model=ov.get("model") or settings.llm_model,
            api_key=ov.get("api_key") or settings.llm_api_key,
            base_url=ov.get("base_url") or settings.llm_base_url,
            streaming=True,
            temperature=0.3,
            max_tokens=2048,
        )
        # Keep the checkpointer/thread id so changing LLM settings in the UI
        # does not erase the conversation history.
        self._agent = create_deep_agent(
            model=model,
            tools=tools,
            system_prompt=_system_prompt(),
            checkpointer=self._checkpointer,
            name="elicitation_demo",
        )
        self._llm_key = key
        return self._agent

    async def close(self) -> None:
        for fut in list(self.pending.values()):
            if not fut.done():
                fut.set_result(("cancel", None))
        self.pending.clear()
        self._agent = None
        self._llm_key = None

    # -- Main loop (streaming) ---------------------------------------------

    async def chat_stream(
        self,
        user_text: str,
        emit: Any,
        llm_override: dict[str, Any] | None = None,
    ) -> None:
        """Run one user turn through DeepAgents; emit SSE events."""
        self._emit = emit
        try:
            agent = await self._ensure_agent(llm_override)
            config = {"configurable": {"thread_id": self._thread_id}}
            stream_input: Any = {
                "messages": [{"role": "user", "content": user_text}]
            }
            # Tool-call chunks are streamed by index, then paired with the
            # ToolMessage for a single Input/Output event pair.
            calls: dict[int, dict[str, Any]] = {}
            emitted_call_ids: set[str] = set()

            for _round in range(MAX_ELICITATION_ROUNDS):
                interrupt_payload: dict[str, Any] | None = None

                async for mode, data in agent.astream(
                    stream_input,
                    config=config,
                    stream_mode=["messages", "updates"],
                ):
                    if mode == "messages":
                        token, _meta = data
                        if isinstance(token, AIMessageChunk):
                            reasoning = (token.additional_kwargs or {}).get(
                                "reasoning_content"
                            )
                            if isinstance(reasoning, str) and reasoning:
                                emit({"type": "reasoning_delta", "text": reasoning})
                            elif isinstance(reasoning, list):
                                for block in reasoning:
                                    text = (
                                        block.get("text", "")
                                        if isinstance(block, dict)
                                        else str(block)
                                    )
                                    if text:
                                        emit(
                                            {"type": "reasoning_delta", "text": text}
                                        )
                            for tc in token.tool_call_chunks or []:
                                idx = tc.get("index")
                                idx = 0 if idx is None else idx
                                state = calls.setdefault(
                                    idx, {"id": "", "name": "", "args": ""}
                                )
                                if tc.get("id"):
                                    state["id"] = tc["id"]
                                if tc.get("name"):
                                    state["name"] += tc["name"]
                                if tc.get("args"):
                                    state["args"] += tc["args"]
                            # Plain text only (tool-call chunks have empty content).
                            if token.content and not token.tool_call_chunks:
                                emit(
                                    {
                                        "type": "assistant_delta",
                                        "text": _text_of(token.content),
                                    }
                                )
                        elif isinstance(token, ToolMessage):
                            call_id = token.tool_call_id or ""
                            if call_id and call_id not in emitted_call_ids:
                                emitted_call_ids.add(call_id)
                                state = next(
                                    (
                                        s
                                        for s in calls.values()
                                        if s["id"] == call_id
                                        and s["name"]
                                    ),
                                    None,
                                )
                                name = token.name or (
                                    state["name"] if state else call_id
                                )
                                raw_args = state["args"] if state else ""
                                try:
                                    args = json.loads(raw_args) if raw_args else {}
                                except ValueError:
                                    args = {}
                                emit(
                                    {
                                        "type": "tool_call",
                                        "id": call_id,
                                        "name": name,
                                        "arguments": args,
                                    }
                                )
                            emit(
                                {
                                    "type": "tool_result",
                                    "id": call_id,
                                    "name": token.name or call_id,
                                    "result": _text_of(token.content),
                                }
                            )
                    elif mode == "updates":
                        for payload in _extract_interrupts(data):
                            interrupt_payload = payload

                if interrupt_payload is None:
                    break

                resume_value = await self._ask_browser(interrupt_payload, emit)
                stream_input = Command(resume=resume_value)
            else:
                emit(
                    {
                        "type": "error",
                        "message": "Exceeded MCP elicitation round limit.",
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
