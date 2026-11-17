"""MCP demo client: plain tool calls + elicitation flows + LLM probe.

Usage:
    uv run python -m mcp_elicitation_demo.demo_client            # interactive prompts
    uv run python -m mcp_elicitation_demo.demo_client --auto     # non-interactive demo
    uv run python -m mcp_elicitation_demo.demo_client --ping-llm # check the LLM endpoint
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv
from mcp import Client, types
from mcp.client.streamable_http import streamable_http_client

load_dotenv()

BASE_URL = os.getenv("LLM_BASE_URL", "")
MODEL = os.getenv("LLM_MODEL", "")


def _print(title: str, body: str = "") -> None:
    print(f"\n=== {title} ===")
    if body:
        print(body)


# ---------------------------------------------------------------------------
# Elicitation handler: answers in --auto mode, asks in interactive mode.
# ---------------------------------------------------------------------------

def make_elicitation_handler(auto: bool):
    """Return an async callback invoked whenever the server elicits input.

    New SDK signature: (context, params: ElicitRequestParams) -> ElicitResult.
    Form-mode params carry `message` + `requested_schema` (a JSON schema dict).
    """

    def _answer(schema: dict) -> dict:
        """Build demo content for a given requested_schema (auto mode)."""
        props = schema.get("properties", {}) if isinstance(schema, dict) else {}
        names = set(props)
        if names == {"name", "party_size", "date", "time"}:
            return {"name": "Alice", "party_size": 4, "date": "2026-12-24", "time": "19:30"}
        if names == {"age", "income", "debt_ratio"}:
            return {"age": 30, "income": 180000.0, "debt_ratio": 0.2}
        if names == {"value"}:  # scalar (bool) response_type is wrapped as {"value": ...}
            return {"value": True}
        return {k: props[k].get("default", "") for k in props}

    def _ask(schema: dict) -> tuple[str, dict | None]:
        """Interactive prompt. Returns (action, content)."""
        if schema.get("properties", {}).keys() == {"value"}:
            raw = input("Confirm? [y/n/cancel]: ").strip().lower()
            if raw.startswith("y"):
                return "accept", {"value": True}
            if raw.startswith("c"):
                return "cancel", None
            return "decline", None
        props = schema.get("properties", {})
        print("Fill in the fields (Enter to skip -> decline):")
        data: dict = {}
        for fname, spec in props.items():
            raw = input(f"  {fname} ({spec.get('description', fname)}): ").strip()
            if not raw:
                continue
            ftype = spec.get("type")
            if ftype == "integer":
                data[fname] = int(raw)
            elif ftype == "number":
                data[fname] = float(raw)
            elif ftype == "boolean":
                data[fname] = raw.lower().startswith(("1", "y", "t"))
            else:
                data[fname] = raw
        return ("accept", data) if data else ("decline", None)

    async def handler(context, params: types.ElicitRequestParams) -> types.ElicitResult:
        if getattr(params, "mode", "form") != "form":
            return types.ElicitResult(action="decline", content=None)
        message = params.message
        schema = params.requested_schema or {}
        print(f"\n[server asks] {message}")
        print(f"[schema]      {schema if schema else 'boolean confirm'}")

        if auto:
            return types.ElicitResult(action="accept", content=_answer(schema))
        action, content = _ask(schema)
        return types.ElicitResult(action=action, content=content)  # type: ignore[arg-type]

    return handler


def make_declining_handler():
    """Always-decline callback: shows the tool's decline branch."""

    async def handler(context, params: types.ElicitRequestParams) -> types.ElicitResult:
        print(f"\n[server asks] {params.message}")
        print("[client]      declining automatically")
        return types.ElicitResult(action="decline", content=None)

    return handler


# ---------------------------------------------------------------------------
# LLM probe: direct OpenAI-compatible call to verify .env credentials.
# ---------------------------------------------------------------------------

def ping_llm() -> None:
    from openai import OpenAI

    if not os.getenv("LLM_API_KEY"):
        _print("LLM probe", "LLM_API_KEY is not set in .env - aborting.")
        sys.exit(1)
    client = OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=BASE_URL)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Reply with exactly: PONG"}],
        max_tokens=2048,
        temperature=0,
    )
    msg = resp.choices[0].message
    _print("LLM probe OK", f"model={resp.model}\nreply={msg.content!r}")


# ---------------------------------------------------------------------------
# Main demo flow.
# ---------------------------------------------------------------------------

async def run(auto: bool, decline: bool = False) -> None:
    url = f"http://{os.getenv('HOST', '127.0.0.1')}:{os.getenv('PORT', '8000')}{os.getenv('MCP_PATH', '/mcp')}"
    _print("Connecting", url)

    callback = make_declining_handler() if decline else make_elicitation_handler(auto)

    # mode="legacy" uses the classic initialize handshake (handshake-era
    # protocol), which keeps the server->client elicitation back-channel.
    # The 2026-07-28 protocol removed it in favor of the input-required
    # retry loop; pinning legacy keeps the classic ctx.elicit() demo simple.
    async with Client(
        streamable_http_client(url),
        elicitation_callback=callback,
        mode="legacy",
    ) as client:
        _print("Initialized", f"server: {client.server_info.name}")

        tools = await client.list_tools()
        _print("Tools", ", ".join(t.name for t in tools.tools))

        # 1) Plain tool calls (no elicitation).
        for name, args in [
            ("add", {"a": 2, "b": 3}),
            ("multiply", {"a": 7, "b": 6}),
            ("greet", {"name": "GLM"}),
            ("dice", {"sides": 20}),
            ("echo", {"text": "plain tool call"}),
            ("now_iso", {}),
        ]:
            res = await client.call_tool(name, args)
            _print(f"tool {name}({args})", res.content[0].text)

        # 2) Elicitation: data request (Booking).
        res = await client.call_tool("book_table", {"city": "Kunming"})
        _print("elicitation book_table", res.content[0].text)

        # 3) Elicitation: boolean confirm.
        res = await client.call_tool("confirm_payment", {"amount": 42.5})
        _print("elicitation confirm_payment", res.content[0].text)

        # 4) Elicitation: structured risk score.
        res = await client.call_tool("risk_score", {"amount": 5000.0})
        _print("elicitation risk_score", res.content[0].text)

        # 5) LLM-backed tool through MCP.
        res = await client.call_tool(
            "extract_entities",
            {"text": "Tim Cook met Satya Nadella in Seattle to discuss AI."},
        )
        _print("tool extract_entities (via LLM)", res.content[0].text)

    _print("Done", "all demo flows finished.")


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP elicitation demo client")
    parser.add_argument("--auto", action="store_true", help="auto-answer elicitations")
    parser.add_argument(
        "--auto-decline", action="store_true", help="auto-DECLINE every elicitation"
    )
    parser.add_argument("--ping-llm", action="store_true", help="only test the LLM endpoint")
    args = parser.parse_args()

    if args.ping_llm:
        ping_llm()
        return
    asyncio.run(run(auto=args.auto, decline=args.auto_decline))


if __name__ == "__main__":
    main()
