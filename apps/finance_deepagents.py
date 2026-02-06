"""
Deep Agent setup for multi-agent financial research.

This module builds a deep financial research agent using the deepagents framework.
It configures:
- A main orchestrator agent responsible for task planning and coordination
- A specialized financial research sub-agent for delegated research tasks
- A secure, per-session filesystem backend for storing research outputs
- SQLite-based checkpointing for recoverable agent execution

Typical usage:
- Initialize the agent with a user_id and thread_id
- Stream responses for long-running, multi-step financial research tasks
"""


from scripts.agent_utils import stream_agent_response
from scripts.deep_prompts import DEEP_RESEARCHER_INSTRUCTIONS, DEEP_ORCHESTRATOR_INSTRUCTIONS
from deepagents.backends import FilesystemBackend
from deepagents import create_deep_agent
from scripts.rag_tools import hybrid_search, live_finance_researcher, think_tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime
import sqlite3
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')
load_dotenv()

# Base directory for all research outputs
RESEARCH_OUTPUT_DIR = os.path.join(os.getcwd(), "research_outputs")


def get_research_backend(user_id, thread_id):
    """
    Create a secure filesystem backend for a specific user session.

    Each (user_id, thread_id) pair gets an isolated output directory to store
    research artifacts generated during agent execution.

    Args:
        user_id: Unique identifier for the user.
        thread_id: Unique identifier for the research session/thread.

    Returns:
        A FilesystemBackend instance with virtual filesystem isolation enabled.
    """

    USER_OUTPUT_DIR = os.path.join(RESEARCH_OUTPUT_DIR, user_id, thread_id)

    os.makedirs(USER_OUTPUT_DIR, exist_ok=True)

    print(f"Writing research files to: {USER_OUTPUT_DIR}")

    backend = FilesystemBackend(
        root_dir=USER_OUTPUT_DIR,
        virtual_mode=True,
    )

    return backend


# Current date injected into researcher instructions for temporal grounding
current_date = datetime.now().strftime("%Y-%m-%d")

# Definition of the delegated financial research sub-agent
research_sub_agent = {
    "name": "financial-research-agent",
    "description": "Delegate financial research to this sub-agent. Give it one specific research task at a time.",
    "system_prompt": DEEP_RESEARCHER_INSTRUCTIONS.format(date=current_date),
    "tools": [hybrid_search, live_finance_researcher, think_tool],
}


# Initialize model
model = ChatGoogleGenerativeAI(model='gemini-3-flash-preview')

# Tools available to the main orchestrator agent
tools = [hybrid_search, live_finance_researcher, think_tool]


def get_deep_agent(user_id, thread_id):
    """
    Initialize and return a deep financial research agent.

    The agent includes:
    - A main orchestrator agent with planning and coordination logic
    - A delegated financial research sub-agent
    - SQLite-based checkpointing for recoverable execution
    - A secure filesystem backend for storing research outputs

    Args:
        user_id: Unique identifier for the user.
        thread_id: Unique identifier for the research session/thread.

    Returns:
        A configured deep research agent ready to handle complex financial queries.
    """

    # SQLite checkpointer for persistent agent memory
    conn = sqlite3.connect(
        'data/deep_agent_finance_researcher.db', check_same_thread=False)
    checkpointer = SqliteSaver(conn=conn)

    backend = get_research_backend(user_id, thread_id)

    # Create the deep agent with memory and secure file backend
    agent = create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=DEEP_ORCHESTRATOR_INSTRUCTIONS,
        subagents=[research_sub_agent],
        checkpointer=checkpointer,  # SQLite memory
        backend=backend,  # Secure filesystem with virtual_mode=True
    )

    return agent


if __name__ == "__main__":
    query = "Compare Apple and Amazon's 2024 revenue and profitability. Present full and detailed report."
    user_id = "kgptalkie"
    thread_id = "session2"

    agent = get_deep_agent(user_id, thread_id)
    stream_agent_response(agent, query, thread_id)
