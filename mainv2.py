import math
import time
import gradio as gr
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

search_tool = DuckDuckGoSearchRun()


@tool
def calculator_tool(expression: str) -> str:
    """Useful ONLY for evaluating mathematical expressions.
    Input must be a valid mathematical string, e.g., '5000000 / 19000000 * 100'.
    Do NOT pass text or natural language here.
    """
    try:
        allowed = {"sqrt": math.sqrt, "pow": pow, "abs": abs}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


@tool
def broken_tool(query: str) -> str:
    """A tool designed to always fail for testing failure cases and circuit breaker."""
    return "Error 404: Server unreachable. Retry in progress..."


tools = [search_tool, calculator_tool, broken_tool]
tools_map = {}
for t in tools:
    tools_map[t.name] = t


def approx_tokens(content) -> int:
    """Rough token estimate. Handles both string and list-based message content."""
    if isinstance(content, list):
        text = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    else:
        text = str(content)
    return int(len(text.split()) * 1.3)


class ResearchAgent:
    def __init__(self, max_turns=5, max_failures=2):
        self.llm = ChatOllama(model="qwen2.5:3b",
                              temperature=0).bind_tools(tools)
        self.system_prompt = SystemMessage(
            content=(
                "You are an autonomous AI Research Assistant.\n\n"
                "STRICT RULES:\n"
                "1. For questions about facts, people, statistics or current events, ALWAYS use 'duckduckgo_search'.\n"
                "2. For mathematical calculations, ALWAYS use 'calculator_tool'.\n"
                "3. Provide a VERY CONCISE, factual final response (1-2 sentences max).\n"
                "4. Do not state that you will use a tool; just call it directly."
            )
        )
        self.max_turns = max_turns
        self.max_failures = max_failures
        self.chat_history = [self.system_prompt]
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def trim_history(self) -> list:
        system = self.chat_history[0]
        turns, current = [], []
        for msg in self.chat_history[1:]:
            current.append(msg)
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                turns.append(current)
                current = []
        if self.max_turns > 0:
            kept = turns[-self.max_turns:]
        else:
            kept = []
        flattened = []
        for t in kept:
            for m in t:
                flattened.append(m)
        flattened.extend(current)
        return [system] + flattened

    def call_tool_safely(self, tool_name: str, tool_args: dict) -> str:
        if tool_name not in tools_map:
            return f"Error: Tool '{tool_name}' does not exist."
        try:
            output = str(tools_map[tool_name].invoke(tool_args))
            return output[:1500]
        except Exception as e:
            return f"Error: Tool '{tool_name}' raised an exception: {e}"

    def save_report(self, query: str, content: str, incomplete: bool = False, tool_log=None):
        filename = "research_report.md"
        status = "INCOMPLETE (step limit reached)" if incomplete else "Completed"
        with open(filename, "a", encoding="utf-8") as f:
            f.write("\n---\n")
            f.write("# AI Research Report\n\n")
            f.write(f"Query: {query}\n\n")
            f.write(f"Status: {status}\n\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Report Content\n\n{content}\n")
            if tool_log:
                f.write("\nRaw Tool Data (Audit Trail)\n\n")
                for i, (name, args, output) in enumerate(tool_log, start=1):
                    f.write(f"{i}. `{name}`\n")
                    f.write(f"- Args: `{args}`\n")
                    f.write(f"- Output: `{output}`\n\n")

    def run(self, user_query: str, max_iterations: int = 5) -> str:
        if user_query.strip().lower() == "cost":
            total_tokens = self.total_prompt_tokens + self.total_completion_tokens
            est_cost_usd = (self.total_prompt_tokens / 1_000_000 * 0.15) + \
                (self.total_completion_tokens / 1_000_000 * 0.60)

            return (
                f"*Session Usage Statistics:\n"
                f" Prompt Tokens: {self.total_prompt_tokens}\n"
                f"*Completion Tokens: {self.total_completion_tokens}\n"
                f" Total Tokens: {total_tokens}\n"
                f" Local API Cost (Ollama): $0.00\n"
                f" Est. Commercial Cloud Cost (GPT-4o-mini): ${est_cost_usd:.6f} USD"
            )

        self.chat_history.append(HumanMessage(content=user_query))
        self.chat_history = self.trim_history()

        tool_failure_counts = {}
        tool_log = []
        step = 0

        while step < max_iterations:
            step += 1

            prompt_text = " ".join([str(m.content) for m in self.chat_history])
            self.total_prompt_tokens += approx_tokens(prompt_text)

            try:
                response = self.llm.invoke(self.chat_history)
            except Exception as e:
                err_msg = f"Model call failed: {e}"
                self.save_report(user_query, err_msg)
                return err_msg

            self.total_completion_tokens += approx_tokens(response.content)

            self.chat_history.append(response)

            if response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    call_id = tool_call.get("id") or f"call_{step}"

                    if tool_failure_counts.get(tool_name, 0) >= self.max_failures:
                        output = f"Error: Tool '{tool_name}' failed repeatedly and is disabled."
                    else:
                        output = self.call_tool_safely(tool_name, tool_args)
                        if output.lower().startswith("error"):
                            tool_failure_counts[tool_name] = tool_failure_counts.get(
                                tool_name, 0) + 1

                    tool_log.append((tool_name, tool_args, output))
                    self.chat_history.append(ToolMessage(
                        content=output, tool_call_id=call_id))
            else:
                final_content = response.content.strip() or "No summary generated."
                self.save_report(user_query, final_content, tool_log=tool_log)
                return final_content

        fallback_res = "Task timed out (step limit reached). Saved gathered logs to report."
        self.save_report(user_query, fallback_res,
                         incomplete=True, tool_log=tool_log)
        return fallback_res


agent = ResearchAgent()


def chat_fn(message, history):
    return agent.run(message)


demo = gr.ChatInterface(
    fn=chat_fn,
    title="LucaGBT-V2",
    description="Autonomous Agent with Tools, Memory & Resiliency.",
    examples=[
        "Who won the latest Champions League?",
        "Calculate sqrt(144) * 25",
        "Test broken tool",
        "cost",
    ],
)

if __name__ == "__main__":
    demo.launch()
