"""
Utilities for streaming agent execution and readable console output.

This module provides helper functions to execute agents in streaming mode
and print intermediate reasoning steps, tool calls, and final outputs
in a human-readable way. It is primarily used for long-running,
multi-step agent workflows such as deep research or graph-based agents.
"""
from langchain.messages import HumanMessage, AIMessage, ToolMessage


def stream_agent_response(agent, query, thread_id="default", user_id=None):
    """
    Execute an agent in streaming mode and print readable execution progress.

    Streams agent messages step by step and prints tool calls, summarized tool
    results, and model outputs for monitoring long-running or multi-step workflows.

    Args:
        agent: Initialized agent instance.
        query: User query or task description.
        thread_id: Execution thread identifier (for checkpointing).
        user_id: Optional user identifier.
    """

    config = {'configurable': {'thread_id': thread_id, 'user_id': user_id}}

    state = {'messages': [HumanMessage(
        query)], 'thread_id': thread_id, 'user_id': user_id}

    for chunk in agent.stream(
        state,
        stream_mode='messages',
        config=config
    ):
        # Extract message from chunk
        message = chunk[0] if isinstance(chunk, tuple) else chunk

        # Handle AI messages with tool calls
        if isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                print(f"\n  Tool Called: {tool_call['name']}")
                print(f"   Args: {tool_call['args']}")
                print()

        # Handle tool responses
        elif isinstance(message, ToolMessage):
            print(f"\n  Tool Result (length: {len(message.text)} chars)")
            print()

        # Handle AI text responses
        elif isinstance(message, AIMessage) and message.text:
            if message.text.strip().startswith("Next Step"):
                continue
            print(message.text, end='', flush=True)
