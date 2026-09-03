# LucaGBT-V2 — Autonomous Research Agent

An autonomous AI research assistant built with **LangChain**, **Ollama**, and **Gradio**. The agent can search the web, run calculations, keep track of conversation history, and gracefully handle failing tools — all through a simple chat interface.

## Features

- **Local LLM inference** via [Ollama](https://ollama.com) (`qwen2.5:3b` by default) — no external API keys required.
- **Tool use / function calling**:
  - `duckduckgo_search` — web search for facts, people, statistics, and current events.
  - `calculator_tool` — safe evaluation of mathematical expressions (restricted `eval` with only `sqrt`, `pow`, `abs` allowed).
  - `broken_tool` — a deliberately failing tool used to test the circuit-breaker logic.
- **Circuit breaker**: a tool that fails repeatedly (default: 2 times) is automatically disabled for the rest of the run to avoid wasting steps.
- **Conversation memory with trimming**: keeps only the last N complete conversation turns (default: 5) to bound the context window sent to the model.
- **Iteration limit**: each query is capped at a maximum number of reasoning/tool-call steps (default: 5) to prevent infinite loops.
- **Session cost tracking**: type `cost` in the chat to see estimated prompt/completion token usage and a projected cost if the same usage had gone through a commercial API (GPT-4o-mini pricing used as a reference).
- **Automatic report logging**: every completed or timed-out query is appended to `research_report.md`, including a full audit trail of tool calls, arguments, and outputs.
- **Gradio chat UI**: simple web interface with example prompts to get started quickly.

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) installed and running locally, with the `qwen2.5:3b` model pulled:
```bash
  ollama pull qwen2.5:3b
```
- Python packages:
```bash
  pip install gradio langchain-community langchain-core langchain-ollama duckduckgo-search
```

## Usage

1. Make sure the Ollama service is running:
```bash
   ollama serve
```
2. Run the app:
```bash
   python mainv2.py
```
3. Open the local Gradio URL printed in the terminal (usually `http://127.0.0.1:7860`).
4. Try one of the example prompts, or type your own query:
   - `Who won the latest Champions League?`
   - `Calculate sqrt(144) * 25`
   - `Test broken tool`
   - `cost` — shows session token usage and estimated cost

## How it works

1. **User query** is added to the agent's chat history, which is trimmed to the last `max_turns` complete turns plus the system prompt.
2. **The LLM decides** whether to answer directly or call a tool (web search or calculator), following strict rules defined in the system prompt (always search for facts, always calculate for math, keep answers concise).
3. **Tool calls** are executed safely; errors are caught and returned as text so the model can react to them. If a tool fails `max_failures` times, it's disabled for the remainder of the run.
4. **The loop continues** — feeding tool outputs back to the model — until the model returns a final answer without further tool calls, or the `max_iterations` step limit is reached.
5. **Every run is logged** to `research_report.md`, including status (`Completed` or `INCOMPLETE`), timestamp, final content, and the full tool call audit trail.

## Configuration

Key parameters can be adjusted in `ResearchAgent.__init__` and `ResearchAgent.run`:

| Parameter | Default | Description |
|---|---|---|
| `max_turns` | 5 | Number of past conversation turns kept in context |
| `max_failures` | 2 | Number of failures before a tool is disabled |
| `max_iterations` | 5 | Max reasoning/tool-call steps per query |
| `llm` model | `qwen2.5:3b` | Ollama model used for inference |

## Notes & Limitations

- `calculator_tool` uses Python's `eval` with a restricted namespace (`sqrt`, `pow`, `abs` only) — it is not a full expression parser and should not be used with untrusted, unsanitized input in production settings.
- Token counts are a **rough approximation** (word count × 1.3), not an exact tokenizer count, so cost estimates are indicative only.
- `research_report.md` grows indefinitely across sessions since entries are always appended, never overwritten.
- The agent runs fully locally except for DuckDuckGo web searches, which require internet access.
