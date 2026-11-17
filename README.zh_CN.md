# mcp-elicitation-demo

基于 **uv + FastAPI + FastMCP + python-dotenv** 的 MCP **Elicitation（引导式输入）** 最小演示，同时包含多个普通 MCP 工具调用，并内置 **网页 Chat UI**，可直接在浏览器测试全部工具。

[English](README.md) | [简体中文](README.zh_CN.md)

- 服务端：FastAPI 应用，挂载 FastMCP **streamable HTTP** 端点 `/mcp`
- 客户端：MCP Python SDK 演示客户端（自动 / 交互两种模式）+ 浏览器 Chat UI
- LLM：任意 OpenAI 兼容接口，凭 `.env` 配置（默认 `zai-org/GLM-5.3-Flash`）

## 工具一览

| 工具 | 类型 | 说明 |
| --- | --- | --- |
| `add`、`multiply`、`greet`、`dice`、`echo`、`now_iso` | 普通工具 | 无用户交互 |
| `book_table` | elicitation | 只引导输入缺失的预订字段（姓名 / 人数 / 日期 / 时间 / 座位类型枚举 / 附加服务多选）；用户消息里已提供的信息会作为参数传入，不再重复询问 |
| `confirm_payment` | elicitation | 是/否 确认 |
| `risk_score` | elicitation | 引导输入年龄 / 收入 / 负债比，输出风控结论 |
| `extract_entities` | 普通工具 + LLM | 调用所配 LLM 抽取实体 |

## 快速开始

```bash
# 1. 复制环境变量模板并填入你的 key
cp .env.example .env

# 2. 安装依赖（自动创建 .venv）
uv sync

# 3. 启动服务
uv run uvicorn src.mcp_elicitation_demo.server:app --port 8000

# 4a. 打开网页 Chat UI
#     http://127.0.0.1:8000/

# 4b. 或运行终端演示客户端
uv run python -m mcp_elicitation_demo.demo_client --auto          # 自动接受
uv run python -m mcp_elicitation_demo.demo_client --auto-decline  # 自动拒绝
uv run python -m mcp_elicitation_demo.demo_client                 # 交互式输入
uv run python -m mcp_elicitation_demo.demo_client --ping-llm      # 测试 LLM 连通性
```

MCP 端点为 `http://127.0.0.1:8000/mcp`（streamable HTTP）；健康检查在 `/healthz`。

## Elicitation 工作原理

1. 客户端创建 MCP `Client` 时注册 `elicitation_callback`。
2. 服务端工具执行中途调用 `ctx.elicit(message=..., response_type=SomeModel)`。
3. SDK 向客户端发送 `elicitation/create` 请求；回调构造 `ElicitResult`，
   `action` 为 `accept` / `decline` / `cancel`，可携带 `content`。
4. 工具判断返回值——`AcceptedElicitation`（含 `.data`）、decline 或 cancel——后继续执行。

## 协议说明（重要）

本演示将 MCP 客户端固定为 `mode="legacy"`（经典 `initialize` 握手，握手期协议版本）。
最新的 MCP 协议（`2026-07-28`）**移除了服务端发起的 elicitation 反向通道**，改为客户端驱动的
"input required" 重试循环（SEP-2322）；fastmcp 4.x 在此类连接上调用 `ctx.elicit()` 会抛出
`elicitation via server-initiated requests is unavailable on 2026-07-28 connections`。
经典握手保留了大多数教程所描述的标准 `ctx.elicit()` 流程。

fastmcp 4.x 两个值得注意的 API 细节：

- `ctx.elicit()` 的参数是 `response_type=`（不是 `schema=`），返回值是类型化结果：
  `AcceptedElicitation` / `DeclinedElicitation` / `CancelledElicitation`。
- 标量 `response_type`（如 `bool`）在线上会被包装成 `{"value": ...}`；
  客户端应答需传 `{"value": True}` 而不是裸布尔值。

## 配置（`.env`）

| 键 | 示例 |
| --- | --- |
| `LLM_API_KEY` | `sk-...` |
| `LLM_BASE_URL` | `https://new-api.app.ynu.edu.cn/v1` |
| `LLM_MODEL` | `zai-org/GLM-5.3-Flash` |

## 架构

```
浏览器 Chat UI ──► POST /api/chat (SSE) ──► ChatSession（GLM function calling 循环）
                     │                          └── 内置 MCP Client ──► /mcp（自连）
                     │                                ├── 普通工具直接执行
                     │                                └── elicitation 工具 → ctx.elicit()
                     │                                      → SSE 事件 → 浏览器弹窗表单
                     │                                      → POST /api/elicitation/respond
                     └── MCP 客户端/其它工具 ──► POST/GET/DELETE /mcp (JSON-RPC + SSE)
FastAPI 同时提供 /、/healthz
```

`server.py` 中沉淀的 FastMCP 4.x 集成要点：

- `mcp.http_app(transport="streamable-http")` 取代了旧的
  `streamable_http_app()`；其 `lifespan` **必须**传给 FastAPI
  （`FastAPI(lifespan=mcp_app.lifespan)`），否则所有请求报
  "Task group is not initialized"。
- `http_app()` 内部已在 `/mcp` 提供 JSON-RPC 路由，所以 FastAPI 挂载点是
  `"/"`，且路由注册必须写在挂载**之前**。

## Chat 桥（chat.py）实现要点

- **真流式**：GLM 走 OpenAI SDK `stream=True`；思考链（`reasoning_content`）与正文
  分别以 `reasoning_delta` / `assistant_delta` 增量推送；工具调用参数按
  `index` 增量拼装。阻塞调用包在 `asyncio.to_thread`，不阻塞事件推送。
- **LLM 覆盖**：`POST /api/chat` 可携带 `llm_override: {base_url, api_key, model}`，
  逐会话覆盖 `.env` 默认值；设置面板存于浏览器 localStorage。
- elicitation 通过 `asyncio.Future` 桥接：SSE 发 `elicitation` 事件给浏览器，
  浏览器提交表单到 `/api/elicitation/respond`，`Future.set_result` 唤醒工具继续。
- SSE 陷阱：结束哨兵必须与事件走**同一条**
  `loop.call_soon_threadsafe(queue.put_nowait, ...)` 通道；若哨兵直接
  `queue.put_nowait`，会抢在已调度事件之前导致流被截断。
- fastmcp `list_tools()` 返回扁平 list；工具 schema 字段是 `input_schema`；
  给 GLM 的 function schema 只保留 `type`/`description`/`enum`。

## Chat UI（首页）

移动端优先的响应式单页应用：

- **思考块**：模型的推理流（如 GLM `reasoning_content`）实时进"深度思考"
  折叠卡，结束后自动收起，可点开回看。
- **工具卡**：每次调用一张卡，`Input` / `Output` 两个 Tab 分开展示参数与结果，
  执行中有 spinner，完成后转绿点。
- **Elicitation 弹窗**：按 schema 动态生成表单——文本/数字输入、
  原生**日期/时间选择器**（`format`）、**枚举单选 pill 组**与
  **多选 checkbox 组**（标题走 SEP-1330 `oneOf [{const,title}]`）、
  布尔确认勾选框，支持 提交 / 拒绝 / 取消。JSON Schema 约束映射为
  原生 HTML 校验（`required`、`pattern`、`min`/`max`、
  `minLength`/`maxLength`、`step`），非法输入无法提交。
- **智能表单**：系统提示注入当天日期，并要求模型把用户已提供的
  信息作为工具参数传入——`book_table` 只弹缺失字段的表单
  （"订一张昆明的餐桌 4 个人" 不会再问人数；信息齐全则完全不弹窗）。
- **设置抽屉**（右上 ⚙️）：可覆盖 base URL / API key / 模型，仅影响当前浏览器
  会话（存 localStorage）；留空回落服务端 `.env` 配置；`GET /api/config`
  提供非密默认值展示。
- 其他：流式打字光标、Enter 发送 + Shift+Enter 换行、自动收缩输入框、
  安全区适配、会话重置（🗑️ 清空）。

## 说明

- 对 `zai-org/GLM-5.3-Flash` 这类推理型模型，`max_tokens` 建议给足
  （>= 1024），否则回复可能在推理阶段被截断。
- 服务端不记录、不存储 API key；凭据只保留在 `.env`
  （或覆盖时存于浏览器 localStorage）。

## 许可证

[MIT](LICENSE)
