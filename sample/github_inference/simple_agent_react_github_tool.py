"""
GitHubModelsInferenceLLM module with ReAct agent implementation.

Author: Ron Webb
Since: 1.0.0
"""

import os
from dotenv import load_dotenv
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from .github_models_inference_chat_model import GitHubModelsInferenceChatModel


@tool
def add_numbers(a: float, b: float) -> float:
    """
    Adds two numbers and returns the result.

    Args:
        a: First number to add
        b: Second number to add

    Returns:
        The sum of a and b
    """
    print(f"Tool: Adding {a} and {b}")
    return a + b


@tool
def multiply_numbers(a: float, b: float) -> float:
    """
    Multiplies two numbers and returns the result.

    Args:
        a: First number to multiply
        b: Second number to multiply

    Returns:
        The product of a and b
    """
    print(f"Tool: Multiplying {a} and {b}")
    return a * b


TOOLS = [add_numbers, multiply_numbers]


if __name__ == "__main__":
    load_dotenv()
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    MODEL_ID = "openai/gpt-4o"

    if not GITHUB_TOKEN:
        raise ValueError("GITHUB_TOKEN is not set in the .env file.")

    print("Initializing enhanced custom chat model...")
    # Use the enhanced GitHubModelsInferenceChatModel with tool calling support
    llm_instance = GitHubModelsInferenceChatModel(
        api_key=GITHUB_TOKEN, model=MODEL_ID, temperature=0
    )

    TOOL_QUESTION = "Calculate 7 + 5 then multiply the result by 2."
    print(f"Tool Question: {TOOL_QUESTION}")

    print("Using create_react_agent with enhanced custom model...")

    try:
        print("Creating ReAct agent...")
        agent = create_react_agent(llm_instance, TOOLS)

        print("Starting agent execution...")
        result = agent.invoke({"messages": [("user", TOOL_QUESTION)]})

        print(f"\n=== Final Answer ===")
        # Extract the final message content from the agent response
        final_message = result["messages"][-1].content
        print(f"{final_message}")
    except Exception as exc:
        print(f"Error occurred: {exc}")
        import traceback

        traceback.print_exc()
