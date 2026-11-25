# mcp-elicitation-demo

A minimal demo of **MCP Elicitation** built with **uv + FastAPI + FastMCP + python-dotenv**, plus several ordinary MCP tools — and a built-in **web chat UI** to try everything from the browser.

[English](README.md) | [简体中文](README.zh_CN.md)

- **Server**: FastAPI app with a FastMCP **streamable HTTP** MCP endpoint at `/mcp`
- **Clients**: MCP Python SDK demo client (auto / interactive) + browser chat UI
- **LLM**: any OpenAI-compatible endpoint via `.env` (default: `zai-org/GLM-5.3-Flash`)

## Tools

| Tool | Kind | Notes |
| --- | --- | --- |
| `add`, `multiply`, `greet`, `dice`, `echo`, `now_iso` | plain | No user interaction |
| `book_table` | elicitation | Asks only for the MISSING booking fields (name / party size / date / time / seating enum / extras multi-select); details already present in the user's message are passed as arguments and never re-asked |
| `confirm_payment` | elicitation | Yes/No confirmation |
| `risk_score` | elicitation | Asks age / income / debt ratio, returns a toy verdict |
| `extract_entities` | plain + LLM | Calls the configured LLM to extract entities |

## Quick start

```bash
# 1. Copy env template and fill in your key
cp .env.example .env

# 2. Install dependencies (creates .venv automatically)
uv sync

# 3. Start the server
uv run mcp-demo            # production-style run
uv run mcp-demo --reload   # dev: auto-reload on code changes
# (equivalent: uv run uvicorn src.mcp_elicitation_demo.server:app --port 8000)

# 4a. Open the chat UI
#     http://127.0.0.1:8000/

# 4b. Or run the terminal demo client
uv run python -m mcp_elicitation_demo.demo_client --auto          # auto-accept elicitations
uv run python -m mcp_elicitation_demo.demo_client --auto-decline  # auto-DECLINE elicitations
uv run python -m mcp_elicitation_demo.demo_client                 # interactive prompts
uv run python -m mcp_elicitation_demo.demo_client --ping-llm      # check LLM endpoint
```

The MCP endpoint is `http://127.0.0.1:8000/mcp` (streamable HTTP); health check at `/healthz`.

### Public URL via cloudflared tunnel

```bash
# one-off quick tunnel (random trycloudflare.com URL)
cloudflared tunnel --url http://localhost:8000

# or a named tunnel: add to ~/.cloudflared/config.yml
#   - hostname: mcp-demo.example.com
#     service: http://localhost:8000
# then run
cloudflared tunnel run <tunnel-name>
```

Note: MCP streamable HTTP works through the tunnel as-is
(POST JSON-RPC + GET SSE on the same `/mcp` path).

## How elicitation works

1. The client registers an `elicitation_callback` when creating the MCP `Client`.
2. A server tool calls `ctx.elicit(message=..., response_type=SomeModel)` mid-execution.
3. The SDK sends an `elicitation/create` request to the client; the callback builds
   an `ElicitResult` with `action` = `accept` / `decline` / `cancel` and optional `content`.
4. The tool inspects the result — an `AcceptedElicitation` (with `.data`), a decline
   or a cancel — and continues accordingly.

## Protocol note (important)

This demo pins the MCP client to `mode="legacy"` (classic `initialize` handshake,
handshake-era protocol version). The newest MCP protocol (`2026-07-28`) **removed the
server-initiated elicitation back-channel** in favor of a client-driven
"input required" retry loop (SEP-2322); fastmcp 4.x raises
`elicitation via server-initiated requests is unavailable on 2026-07-28 connections`
when `ctx.elicit()` is called on such a connection. The legacy handshake keeps the
classic `ctx.elicit()` flow that most tutorials describe.

Two fastmcp 4.x API details worth knowing:

- `ctx.elicit()` takes `response_type=` (not `schema=`) and returns typed results:
  `AcceptedElicitation` / `DeclinedElicitation` / `CancelledElicitation`.
- Scalar `response_type`s (e.g. `bool`) are wrapped as `{"value": ...}` on the wire;
  the client must answer `{"value": true}` rather than a bare boolean.

## Web chat UI

The landing page is a mobile-first, single-file chat app:

- **Thinking block** — the model's reasoning stream (e.g. GLM `reasoning_content`)
  flows into a collapsible "thinking" card that auto-collapses when done.
- **Tool cards** — one card per call with separate **Input / Output** tabs,
  a spinner while running and a green dot when finished.
- **Elicitation modal** — the form is generated from the requested schema:
  text / number inputs, native **date/time pickers** (`format`),
  **radio groups for enums** and **checkbox groups for multi-select
  arrays** (titles via SEP-1330 `oneOf [{const,title}]`), a checkbox for
  boolean confirms, with submit / decline / cancel actions.
  JSON-Schema constraints map to native HTML validation (`required`,
  `pattern`, `min`/`max`, `minLength`/`maxLength`, `step`); invalid
  input blocks submit.
- **Smart forms** — the system prompt carries today's date and instructs
  the model to pass already-known details as tool arguments, so
  `book_table` elicits ONLY the missing fields ("book a table in
  Kunming for 4" never asks for party size again; with full info no
  modal appears at all).
- **Settings drawer** (top-right ⚙️) — override base URL / API key / model
  per browser session (stored in `localStorage`); empty fields fall back to the
  server `.env` defaults. `GET /api/config` exposes the non-secret defaults.
- Extras: streamed tokens with a typing cursor, Enter to send / Shift+Enter for
  newline, auto-growing composer, safe-area insets, and session reset.

## Configuration (`.env`)

| Key | Example |
| --- | --- |
| `LLM_API_KEY` | `sk-...` |
| `LLM_BASE_URL` | `https://new-api.app.ynu.edu.cn/v1` |
| `LLM_MODEL` | `zai-org/GLM-5.3-Flash` |

## Architecture

```
Browser chat UI ─► POST /api/chat (SSE) ─► ChatSession (DeepAgents graph on LangGraph)
                     │                          └── MCPAdapter ──► /mcp (self)
                     │                                ├── plain tools run directly
                     │                                └── elicitation → LangGraph interrupt()
                     │                                      → SSE event → browser modal form
                     │                                      → POST /api/elicitation/respond
                     │                                      → Command(resume=...) → same graph run
Terminal demo client ─► POST/GET/DELETE /mcp  (JSON-RPC + SSE)
FastAPI also serves / (chat UI) and /healthz
```

### Branches

| Branch | Chat bridge | Agent loop | Elicitation bridge |
| --- | --- | --- | --- |
| `main` | `chat.py` | hand-rolled OpenAI function calling | `asyncio.Future` |
| `agentscope` | `chat_agentscope.py` | AgentScope `Agent.reply_stream` | MCP SDK callback + `Future` |
| `deepagent` (this branch) | `chat_deepagent.py` | DeepAgents SDK on LangGraph | LangGraph `interrupt()` / `Command(resume=...)` |

All branches expose the identical SSE event contract, so the browser UI is
unchanged. This branch replaces the hand-written loop with
`create_deep_agent(model=..., tools=...)`; MCP tools come from the modern
`langchain.mcp.MCPAdapter` (`langchain[mcp]>=1.4`, beta). The adapter converts a
server's `InputRequiredResult` into a LangGraph `MCPElicitationInterrupt` —
each form request is emitted to the browser, and the submitted answers are
returned as `Command(resume={"responses": ...})`, resuming the **same** graph
run via an `InMemorySaver` checkpointer and a stable `thread_id`.

DeepAgents bridge (`chat_deepagent.py`) implementation notes:

- **Streaming** — `agent.astream(..., stream_mode=["messages", "updates"])`:
  `AIMessageChunk` yields `reasoning_delta` / `assistant_delta`, tool-call
  chunks are accumulated by `index` and paired with their `ToolMessage`
  for one `tool_call` + `tool_result` pair.
- **LLM overrides** — `POST /api/chat` accepts
  `llm_override: {base_url, api_key, model}`; rebuilding the agent keeps the
  checkpointer and thread id, so changing settings does not erase history.
- **Elicitation** — the SSE `elicitation` event carries the adapted
  `requested_schema`; the browser POSTs to `/api/elicitation/respond`, which
  resolves the pending future and lets the graph resume from its interrupt.
- **SSE pitfall** — the end-of-stream sentinel must go through the same
  `loop.call_soon_threadsafe(queue.put_nowait, ...)` channel as events; a direct
  `queue.put_nowait` overtakes already-scheduled events and truncates the stream.
- `langchain.mcp` is in beta, so a `LangChainBetaWarning` on import is expected.

FastMCP 4.x integration notes baked into `server.py`:

- `mcp.http_app(transport="streamable-http")` replaces the old
  `streamable_http_app()`; its `lifespan` **must** be passed to FastAPI
  (`FastAPI(lifespan=mcp_app.lifespan)`) or every request fails with
  "Task group is not initialized".
- `http_app()` already serves the JSON-RPC route at `/mcp`, so the FastAPI
  mount target is `"/"` and route registration must come **before** the mount.

## Notes

- Keep `max_tokens` generous (>= 1024) for reasoning-style models such as
  `zai-org/GLM-5.3-Flash`; otherwise replies can be cut off mid-reasoning.
- The server never logs or stores the API key; it stays in `.env`
  (or in the browser's `localStorage` when overridden).

## License

[MIT](LICENSE)
