"""
LangGraph-based multi-agent financial research pipeline.

This module implements a three-stage deep research workflow using LangGraph
and LangChain agents:

1. Orchestrator: Plans the research, splits the task into themes, and coordinates execution
2. Researcher: Performs focused research for a single thematic question and writes results to disk
3. Editor: Synthesizes all research artifacts into a final report

The system uses:
- Shared agent state (DeepAgentState) with user/thread isolation
- File-based artifacts for auditability and reproducibility
- SQLite checkpointing for recoverable multi-step execution
"""

from langchain_core.utils.function_calling import convert_to_openai_tool
from scripts.prompts import (
    ORCHESTRATOR_PROMPT,
    RESEARCHER_PROMPT,
    EDITOR_PROMPT
)
from scripts.file_tools import (
    DeepAgentState,
    ls,
    read_file,
    write_file,
    cleanup_files,
    generate_hash,
    _disk_path
)
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langchain_core.tools import tool, InjectedToolCallId
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from scripts.rag_tools import hybrid_search, live_finance_researcher
from langgraph.checkpoint.sqlite import SqliteSaver
from scripts.agent_utils import stream_agent_response
import sqlite3
from typing import Annotated
from dotenv import load_dotenv
import traceback
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', streaming=False)

# SQLite checkpointing for recoverable LangGraph execution
conn = sqlite3.connect("data/deep_finance_researcher.db",
                       check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# Tools available for research-related agents
tools = [ls, write_file, read_file, hybrid_search, live_finance_researcher]


# Optional schema inspection to validate tool contracts
for t in tools:
    try:
        schema = t.args_schema.model_json_schema()
        props = schema.get("properties", {})
        print("\nTOOL:", t.name)
        print("properties keys:", list(props.keys()))
        if "config" in props:
            print("!!! config property =", props["config"])
    except Exception as e:
        print("\nTOOL:", getattr(t, "name", t))
        print("SCHEMA BUILD FAILED:", e)

# Researcher agent: performs focused research for one thematic question
researcher_agent = create_agent(
    model=llm,
    tools=[ls, write_file, read_file, hybrid_search, live_finance_researcher],
    system_prompt=RESEARCHER_PROMPT,
    state_schema=DeepAgentState
)

# Editor agent: synthesizes all research artifacts into a final report
editor_agent = create_agent(
    model=llm,
    tools=[ls, write_file, read_file, cleanup_files],
    system_prompt=EDITOR_PROMPT,
    state_schema=DeepAgentState
)


@tool
def write_research_plan(
    thematic_questions: list[str],
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],

):
    """
    Write the high-level research plan with major thematic questions.

    Args:
        thematic_questions: List of 3-5 major thematic questions
        state: Injected agent state
        tool_call_id: Tool call ID

    Returns:
        Command with ToolMessage confirming the plan was written
    """
    content = "# Research Plan\n\n"

    content = content + "## User Query\n"
    content = content + state["messages"][-1].text + "\n\n"

    content = content + "## Thematic Questions\n\n"
    for i, question in enumerate(thematic_questions, 1):
        content = content + f"{i}. {question}\n"

    path = _disk_path(state, "research_plan.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    msg = f"[RESEARCH PLAN WRITTEN] research_plan.md with {len(thematic_questions)} thematic questions"
    return Command(update={"messages": [ToolMessage(msg, tool_call_id=tool_call_id)]})


@tool
def run_researcher(
    theme_id: int,
    thematic_question: str,
    state: Annotated[DeepAgentState, InjectedState],
    max_retries: int = 2
):
    """
    Run a single Research agent for ONE thematic question.

    Args:
        theme_id: The theme number (1, 2, 3, etc.)
        thematic_question: The specific thematic question to research
        state: Injected agent state
        max_retries: Number of retry attempts

    Returns:
        Status string for the orchestrator
    """

    file_hash = generate_hash(f"{theme_id}_{thematic_question}")

    ai_message_instruction = f"""[THEME {theme_id}] {thematic_question}

                        Save research to: researcher/{file_hash}_theme.md
                        Save sources to: researcher/{file_hash}_sources.txt
                        """

    sub_state: DeepAgentState = {
        "messages": state["messages"] + [AIMessage(ai_message_instruction)],
        "user_id": state.get("user_id"),
        "thread_id": state.get("thread_id"),
    }

    for attempt in range(max_retries + 1):
        try:
            researcher_agent.invoke(sub_state)
            return f"✓ Theme {theme_id} research completed (hash: {file_hash})"
        except Exception as e:
            print(f"[Researcher Error] attempt={attempt} error={e}")
            traceback.print_exc()

    return f"✗ Theme {theme_id} failed after {max_retries + 1} attempts"


@tool
def run_editor(state: Annotated[DeepAgentState, InjectedState]) -> str:
    """
    Run the Editor agent to synthesize all research into final report.

    Args:
        state: Injected agent state

    Returns:
        Status string
    """
    sub_state: DeepAgentState = {
        "messages": [HumanMessage(content="Read research_plan.md and all files in the researcher/ folder, then synthesize everything into a comprehensive report.md file.")],
        "user_id": state.get("user_id"),
        "thread_id": state.get("thread_id"),
    }
    editor_agent.invoke(sub_state)
    return "Editor completed. Final report is written to report.md."


# Orchestrator agent coordinating the full research pipeline
orchestrator_agent = create_agent(
    model=llm,
    tools=[write_research_plan, run_researcher, run_editor, cleanup_files],
    system_prompt=ORCHESTRATOR_PROMPT,
    state_schema=DeepAgentState,
    checkpointer=checkpointer
)


if __name__ == "__main__":
    agent = orchestrator_agent
    query = "Do a detailed analysis of Amazon's financial performance in 2023 and 2024"

result = agent.invoke(
    {"messages": [HumanMessage(query)], "user_id": "user_212"},
    config={
        "configurable": {"thread_id": "thread_003"},
        "recursion_limit": 80,
    },
)
print(result)
