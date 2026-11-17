"""MCP Elicitation demo server built with FastAPI + FastMCP (streamable HTTP).

Run with:
    uv run uvicorn src.mcp_elicitation_demo.server:app --port 8000

Endpoints:
    /mcp    MCP streamable HTTP endpoint (POST = JSON-RPC, GET = SSE,
            DELETE = session teardown), served by FastMCP http_app()
    /       chat UI for testing tools in the browser
    /healthz                    health check
    POST /api/chat              SSE stream of one assistant turn
    POST /api/elicitation/respond  browser answer to an elicitation form
    POST /api/session/{id}/reset   drop conversation history
    GET  /api/config            non-secret LLM defaults for the settings UI
"""

from __future__ import annotations

import asyncio
import json
import random
from enum import Enum
from typing import Annotated, Any, Literal

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastmcp import Context, FastMCP
from fastmcp.server.elicitation import AcceptedElicitation
from openai import OpenAI
from pydantic import BaseModel, BeforeValidator, Field, StringConstraints, create_model

from . import chat as chat_bridge
from .settings import settings

_STATIC_DIR = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# App + MCP wiring: one FastAPI app, MCP mounted at settings.path (/mcp).
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="elicitation-demo",
    instructions=(
        "Demo server. Regular tools need no input from the client. "
        "Elicitation tools call ctx.elicit() to ask the user for confirmation "
        "or extra data, then continue."
    ),
)

mcp_app = mcp.http_app(transport="streamable-http")  # serves JSON-RPC at /mcp

# FastAPI must run FastMCP's lifespan so the session manager task group starts.
# NOTE: mount "/" at the very bottom of this file, after all routes are defined,
# otherwise the mount swallows them.
app = FastAPI(title="MCP Elicitation Demo", version="0.1.0", lifespan=mcp_app.lifespan)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Chat bridge API: the browser talks to the LLM; the LLM calls MCP tools.
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str
    llm_override: dict[str, Any] | None = None


class ElicitRespondRequest(BaseModel):
    session_id: str = "default"
    id: str
    action: Literal["accept", "decline", "cancel"]
    content: Any = None


@app.post("/api/chat")
async def api_chat(req: ChatRequest) -> StreamingResponse:
    """One assistant turn as an SSE stream of typed events."""

    async def gen():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        _SENTINEL = object()

        def emit(event: dict) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        sess = chat_bridge.get_session(req.session_id)

        async def run():
            try:
                await sess.chat_stream(req.message, emit, req.llm_override)
            finally:
                # IMPORTANT: the sentinel must go through the same
                # call_soon_threadsafe channel as emit(); a direct
                # queue.put_nowait here can overtake already-scheduled
                # events and truncate the SSE stream.
                loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

        task = asyncio.create_task(run())

        while True:
            item = await queue.get()
            if item is _SENTINEL:
                break
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        await task

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/session/{session_id}/reset")
async def api_session_reset(session_id: str) -> dict[str, bool]:
    """Drop the conversation history and close the MCP connection."""
    await chat_bridge.cleanup_session(session_id)
    return {"ok": True}


@app.get("/api/config")
async def api_config() -> dict[str, str]:
    """Non-secret defaults for the settings UI."""
    return {"base_url": settings.llm_base_url, "model": settings.llm_model}


@app.post("/api/elicitation/respond")
async def api_elicitation_respond(req: ElicitRespondRequest) -> dict[str, bool]:
    sess = chat_bridge.get_session(req.session_id)
    ok = sess.respond_elicitation(req.id, req.action, req.content)
    if not ok:
        raise HTTPException(status_code=404, detail="elicitation not found or already answered")
    return {"ok": True}


@app.get("/", response_class=FileResponse, include_in_schema=False)
def index() -> FileResponse:
    """Serve the chat UI from the bundled static directory."""
    return FileResponse(_STATIC_DIR / "index.html", media_type="text/html")


# ---------------------------------------------------------------------------
# Reusable Pydantic schemas for elicitation requests.
# ---------------------------------------------------------------------------

class Booking(BaseModel):
    """Full booking shape. The tool elicits ONLY the missing fields,
    so the browser form shrinks to what the user has not provided."""

    name: str = Field(description="Guest name", min_length=1)
    party_size: int = Field(ge=1, le=20, description="Number of guests")
    date: Annotated[str, StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}$")] = Field(
        description="Booking date", json_schema_extra={"format": "date"}
    )
    time: Annotated[str, StringConstraints(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")] = Field(
        description="Booking time (24h)", json_schema_extra={"format": "time"}
    )
    seating: Seating = Field(description="Seating type")
    extras: list[Extra] = Field(
        default_factory=list, description="Optional extras (multi-select)"
    )


class Seating(str, Enum):
    """Where to sit - rendered as a radio/select group."""

    hall = "hall"
    private = "private"
    patio = "patio"


Extra = Literal["highchair", "birthday", "window", "vegan"]

# UI-facing titles for enum values (SEP-1330 oneOf {const,title} pattern).
SEATING_TITLES = {"hall": "大厅", "private": "包间", "patio": "露台"}
EXTRA_TITLES = {
    "highchair": "儿童椅",
    "birthday": "生日布置",
    "window": "靠窗",
    "vegan": "素食",
}

# Field specs shared by the dynamic per-missing-field schema builder.
_DATE_STR = Annotated[str, StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}$")]
_TIME_STR = Annotated[str, StringConstraints(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")]
_BOOKING_FIELD_SPECS: dict[str, tuple[Any, dict[str, Any]]] = {
    "name": (str, Field(description="Guest name", min_length=1)),
    "party_size": (int, Field(ge=1, le=20, description="Number of guests")),
    "date": (
        _DATE_STR,
        Field(description="Booking date", json_schema_extra={"format": "date"}),
    ),
    "time": (
        _TIME_STR,
        Field(description="Booking time (24h)", json_schema_extra={"format": "time"}),
    ),
    "seating": (
        Seating,
        Field(description="Seating type"),
    ),
    "extras": (
        list[Extra],
        Field(
            default_factory=list, description="Optional extras (multi-select)"
        ),
    ),
}


def _titled_enum_extra(titles: dict[str, str]) -> dict[str, Any]:
    """SEP-1330 oneOf [{const,title}] hint for a STRING enum field."""
    return {"oneOf": [{"const": k, "title": v} for k, v in titles.items()]}


def _titled_items_extra(titles: dict[str, str]) -> dict[str, Any]:
    """oneOf [{const,title}] hint for ARRAY fields - it must live in
    `items`, an array-level oneOf would match the list itself."""
    return {
        "items": {
            "type": "string",
            "oneOf": [{"const": k, "title": v} for k, v in titles.items()],
        }
    }


def _coerce_list(v: Any) -> Any:
    """LLMs sometimes stream list args as a JSON string - coerce."""
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return [e.strip() for e in v.split(",") if e.strip()]
    return v


# lenient list annotation for LLM-facing tool parameters
LenientList = Annotated[list[str], BeforeValidator(_coerce_list)]


def _booking_model_for(missing: list[str]) -> type[BaseModel]:
    """Pydantic model containing ONLY the fields still missing.
    ctx.elicit() needs a *type* here - a bare dict would be parsed as
    fastmcp's dict-enum shorthand syntax."""
    if not missing:
        return create_model("BookingEmpty")
    fields: dict[str, tuple[Any, Any]] = {}
    for k in missing:
        ftype, fmeta = _BOOKING_FIELD_SPECS[k]
        if k == "seating":
            fmeta = Field(
                description="Seating type",
                json_schema_extra=_titled_enum_extra(SEATING_TITLES),
            )
        elif k == "extras":
            fmeta = Field(
                default_factory=list,
                description="Optional extras (multi-select)",
                json_schema_extra=_titled_items_extra(EXTRA_TITLES),
            )
        fields[k] = (ftype, fmeta)
    return create_model("BookingPartial", **fields)  # type: ignore[arg-type]


class Risk(BaseModel):
    """Data we elicit for a risk score."""

    age: int = Field(ge=18, le=100, description="Age in years")
    income: float = Field(ge=0, description="Annual income in CNY")
    debt_ratio: float = Field(ge=0, le=1, description="Debt / income, 0..1")


# ---------------------------------------------------------------------------
# Plain tools (no elicitation) - the "ordinary" MCP tool calls.
# ---------------------------------------------------------------------------

@mcp.tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@mcp.tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@mcp.tool
def greet(name: str) -> str:
    """Return a friendly greeting."""
    return f"Hello, {name}!"


@mcp.tool
def dice(sides: int = 6) -> int:
    """Roll a dice with the given number of sides."""
    return random.randint(1, sides)


@mcp.tool
def echo(text: str) -> str:
    """Echo the input text back."""
    return text


@mcp.tool
def now_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Elicitation tools - ask the user mid-call via ctx.elicit().
# ---------------------------------------------------------------------------

@mcp.tool
async def book_table(
    city: str,
    ctx: Context,
    name: str | None = None,
    party_size: int | None = None,
    date: str | None = None,
    time: str | None = None,
    seating: str | None = None,
    extras: LenientList | None = None,
) -> str:
    """Book a table.

    Pass any booking details the user already provided (name, party_size,
    date YYYY-MM-DD, time HH:MM, seating: hall|private|patio, extras:
    highchair|birthday|window|vegan); only the MISSING fields are elicited,
    so the browser form shrinks accordingly. If everything is known,
    no form appears at all.
    """
    known: dict[str, Any] = {}
    if name is not None:
        known["name"] = name
    if party_size is not None:
        known["party_size"] = party_size
    if date is not None:
        known["date"] = date
    if time is not None:
        known["time"] = time
    if seating is not None:
        known["seating"] = seating
    if extras:
        known["extras"] = list(extras)

    missing = [k for k in _BOOKING_FIELD_SPECS if k not in known]
    if missing:
        result = await ctx.elicit(
            message=(
                f"Booking a table in {city}"
                + (f" for {party_size} guests" if party_size else "")
                + (f" on {date}" if date else "")
                + (f" at {time}" if time else "")
                + (f" ({seating})" if seating else "")
                + f". Still need: {', '.join(missing)}"
            ),
            response_type=_booking_model_for(missing),
        )
        if not isinstance(result, AcceptedElicitation):
            past = {"decline": "declined", "cancel": "cancelled"}.get(
                result.action, result.action
            )
            return f"Booking {past} by user."
        known.update(result.data or {})

    b = Booking(**known)
    code = f"BK-{random.randint(1000, 9999)}"
    extras_txt = ", ".join(b.extras) if b.extras else "none"
    return (
        f"Table booked in {city}: {b.party_size} guests, "
        f"{b.date} {b.time}, {b.seating.value} seating, "
        f"extras: {extras_txt}, under '{b.name}'. Confirmation {code}."
    )


@mcp.tool
async def confirm_payment(amount: float, ctx: Context) -> str:
    """Elicits a yes/no confirmation before (simulated) payment."""
    result = await ctx.elicit(
        message=f"Confirm payment of {amount:.2f} CNY?",
        response_type=bool,  # builtin types are allowed; a yes/no widget is shown
    )
    if isinstance(result, AcceptedElicitation) and result.data is True:
        return f"Payment of {amount:.2f} CNY completed."
    if result.action == "cancel":
        return f"Payment dialog for {amount:.2f} CNY dismissed."
    return f"Payment of {amount:.2f} CNY declined; nothing was charged."


@mcp.tool
async def risk_score(amount: float, ctx: Context) -> str:
    """Elicits age, income and debt ratio, then returns a toy risk decision."""
    result = await ctx.elicit(
        message="Risk assessment needs a few details:",
        response_type=Risk,
    )
    if not isinstance(result, AcceptedElicitation):
        past = {"decline": "declined", "cancel": "cancelled"}.get(result.action, result.action)
        return f"Risk assessment {past} by user."

    r: Risk = result.data
    score = 0.0
    if r.age < 25:
        score += 1
    if r.income < amount * 12:
        score += 2
    score += r.debt_ratio * 5

    verdict: Literal["approved", "manual-review", "declined"] = (
        "approved" if score < 2 else "manual-review" if score < 4 else "declined"
    )
    return (
        f"Risk score {score:.1f} -> {verdict} "
        f"(age={r.age}, income={r.income:.0f}, debt_ratio={r.debt_ratio:.2f}, amount={amount:.2f})"
    )


# ---------------------------------------------------------------------------
# LLM-backed tool - uses the configured OpenAI-compatible endpoint.
# ---------------------------------------------------------------------------

@mcp.tool
def extract_entities(text: str) -> dict[str, Any]:
    """Extract person names, organizations and places from text via the LLM."""
    if not settings.llm_ready:
        return {"error": "LLM_API_KEY not set; configure .env first."}

    client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract entities from the user text. Reply ONLY with JSON: "
                    '{"persons": [...], "orgs": [...], "places": [...]}. '
                    "Use [] when a category has no entity. No prose."
                ),
            },
            {"role": "user", "content": text},
        ],
        max_tokens=2048,
        temperature=0,
    )
    content = resp.choices[0].message.content or "{}"
    content = content[content.find("{") : content.rfind("}") + 1]
    try:
        import json

        return json.loads(content)
    except (ValueError, TypeError):
        return {"error": "Model returned non-JSON content", "raw": content[:200]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)

# Mount LAST so FastAPI routes (/, /healthz) keep priority; mcp_app serves /mcp.
app.mount("/", mcp_app)
